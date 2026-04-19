/*
* Name: Ahilesh Vadivel
* Date: February 20th 2026
* 
 * Project 3 - Object Recognition Pipeline
 * Tasks 1 through 8:
 *   Task 1 - Thresholding (ISODATA + custom HSV threshold, from scratch)
 *   Task 2 - Morphological cleanup (Opening + Closing)
 *   Task 3 - Connected components segmentation
 *   Task 4 - Feature extraction (axis, OBB, 5 invariant features)
 *   Task 5 - Training: collect labeled feature vectors, save to CSV
 *   Task 6 - Classification: scaled Euclidean nearest-neighbor
 *   Task 7 - Evaluation: confusion matrix
 *   Task 9 - One-shot embedding classification via ResNet18
 *
 * Controls:
 *   'q' / ESC  - quit
 *   's'        - save images
 *   't'        - train hand-crafted features (Task 5/6)
 *   'r'        - train embedding (Task 8) — label current object, save embedding
 *   'e'        - evaluate: record true label vs prediction for confusion matrix
 *   'p'        - print and save confusion matrix
 *   '+' / '='  - raise threshold
 *   '-' / '_'  - lower threshold
 *
 * Requirements:
 *   - resnet18.onnx must be in the same folder as the executable
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <iomanip>

 // STRUCTS
struct RegionData {
    int         label;
    int         area;
    double      cx, cy;
    int         x, y, width, height;
    cv::Vec3b   color;
};

struct FeatureData {
    double      alpha;
    double      hwRatio;
    double      percentFilled;
    double      eccentricity;
    double      huMoment1;
    double      huMoment2;
    cv::Point2f obbCorners[4];
    cv::Point2f axisStart, axisEnd;
    // OBB extents along major/minor axes — needed for ROI extraction in Task 8
    double      Dmin, Dmax;   // major axis extents
    double      dmin, dmax;   // minor axis extents
};

struct TrainingEntry {
    std::string label;
    double      hwRatio;
    double      percentFilled;
    double      eccentricity;
    double      huMoment1;
    double      huMoment2;
};

struct ClassificationResult {
    std::string label;
    double      distance;
    bool        isUnknown;
};

// Stores one labeled embedding for Task 8
struct EmbeddingEntry {
    std::string        label;
    std::vector<float> embedding; // 512-dim ResNet18 avgpool output
};

// TASK 7 - CONFUSION MATRIX
struct ConfusionMatrix {
    std::vector<std::string>                      labels;
    std::map<std::string, std::map<std::string, int>> counts;
    int total = 0, correct = 0;

    void initLabels(const std::vector<TrainingEntry>& db) {
        for (const auto& e : db)
            if (std::find(labels.begin(), labels.end(), e.label) == labels.end())
                labels.push_back(e.label);
        std::sort(labels.begin(), labels.end());
    }

    void record(const std::string& trueLabel, const std::string& predicted) {
        if (std::find(labels.begin(), labels.end(), trueLabel) == labels.end()) {
            labels.push_back(trueLabel);
            std::sort(labels.begin(), labels.end());
        }
        counts[trueLabel][predicted]++;
        total++;
        if (trueLabel == predicted) correct++;
    }

    void print() const {
        if (labels.empty()) { std::cout << "No evaluation data yet.\n"; return; }
        int colW = 10;
        for (const auto& l : labels) colW = std::max(colW, (int)l.size() + 2);
        std::cout << "\n===== CONFUSION MATRIX =====\n";
        std::cout << "Rows=True | Cols=Predicted\n\n";
        std::cout << std::string(colW, ' ');
        for (const auto& l : labels) std::cout << std::setw(colW) << l;
        std::cout << "\n" << std::string(colW * ((int)labels.size() + 1), '-') << "\n";
        for (const auto& tl : labels) {
            std::cout << std::setw(colW) << tl;
            for (const auto& pl : labels) {
                int cnt = 0;
                auto ri = counts.find(tl);
                if (ri != counts.end()) {
                    auto ci = ri->second.find(pl);
                    if (ci != ri->second.end()) cnt = ci->second;
                }
                std::cout << std::setw(colW) << cnt;
            }
            std::cout << "\n";
        }
        std::cout << "\nTotal: " << total;
        if (total > 0)
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(1)
            << (100.0 * correct / total) << "%";
        std::cout << "\n============================\n\n";
    }

    void saveToFile(const std::string& fn) const {
        std::ofstream f(fn);
        if (!f.is_open()) return;
        f << "True\\Predicted";
        for (const auto& l : labels) f << "," << l;
        f << "\n";
        for (const auto& tl : labels) {
            f << tl;
            for (const auto& pl : labels) {
                int cnt = 0;
                auto ri = counts.find(tl);
                if (ri != counts.end()) {
                    auto ci = ri->second.find(pl);
                    if (ci != ri->second.end()) cnt = ci->second;
                }
                f << "," << cnt;
            }
            f << "\n";
        }
        if (total > 0)
            f << "Accuracy," << (100.0 * correct / total) << "%\n";
        std::cout << "Confusion matrix saved to " << fn << "\n";
    }
};

// COLOR PALETTE
static const std::vector<cv::Vec3b> COLOR_PALETTE = {
    {255,80,80},{80,255,80},{80,80,255},{255,255,80},{255,80,255},
    {80,255,255},{255,165,0},{180,0,255},{0,200,150},{255,200,150},
};

// TASK 1 - ISODATA THRESHOLD (from scratch)
int isodataThreshold(const cv::Mat& grayImg)
{
    std::vector<int> samples;
    samples.reserve(grayImg.rows * grayImg.cols / 16);
    for (int r = 0; r < grayImg.rows; r += 4)
        for (int c = 0; c < grayImg.cols; c += 4)
            samples.push_back((int)grayImg.at<uchar>(r, c));
    double m0 = 80.0, m1 = 200.0;
    for (int i = 0; i < 50; i++) {
        double s0 = 0, s1 = 0; int c0 = 0, c1 = 0;
        for (int v : samples) {
            if (std::abs(v - m0) <= std::abs(v - m1)) { s0 += v; c0++; }
            else { s1 += v; c1++; }
        }
        double n0 = (c0 > 0) ? s0 / c0 : m0, n1 = (c1 > 0) ? s1 / c1 : m1;
        if (std::abs(n0 - m0) < 0.5 && std::abs(n1 - m1) < 0.5) break;
        m0 = n0; m1 = n1;
    }
    return (int)((m0 + m1) / 2.0);
}

// TASK 1 - CUSTOM THRESHOLD (from scratch)
cv::Mat customThreshold(const cv::Mat& hsvImg, int valT, int satT)
{
    cv::Mat bin(hsvImg.rows, hsvImg.cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < hsvImg.rows; r++)
        for (int c = 0; c < hsvImg.cols; c++) {
            uchar s = hsvImg.at<cv::Vec3b>(r, c)[1];
            uchar v = hsvImg.at<cv::Vec3b>(r, c)[2];
            if (v<valT || s>satT) bin.at<uchar>(r, c) = 255;
        }
    return bin;
}

// TASK 2 - MORPHOLOGICAL CLEANUP
cv::Mat morphologicalCleanup(const cv::Mat& binary, int ks = 3)
{
    if (ks % 2 == 0) ks++;
    cv::Size kSz(ks, ks);
    cv::Mat k4 = cv::getStructuringElement(cv::MORPH_CROSS, kSz);
    cv::Mat k8 = cv::getStructuringElement(cv::MORPH_RECT, kSz);
    cv::Mat t, op, di, cl;
    cv::erode(binary, t, k4); cv::dilate(t, op, k8);
    cv::dilate(op, di, k4);   cv::erode(di, cl, k8);
    return cl;
}

// TASK 3 - SEGMENT REGIONS
std::vector<RegionData> segmentRegions(const cv::Mat& cleaned,
    cv::Mat& regionMap,
    int minArea = 500, int maxReg = 5)
{
    cv::Mat lm, st, ct;
    int nl = cv::connectedComponentsWithStats(cleaned, lm, st, ct, 8, CV_32S);
    int W = cleaned.cols, H = cleaned.rows;
    std::vector<RegionData> valid;
    for (int l = 1; l < nl; l++) {
        int a = st.at<int>(l, cv::CC_STAT_AREA);
        int bx = st.at<int>(l, cv::CC_STAT_LEFT), by = st.at<int>(l, cv::CC_STAT_TOP);
        int bw = st.at<int>(l, cv::CC_STAT_WIDTH), bh = st.at<int>(l, cv::CC_STAT_HEIGHT);
        if (a < minArea || bx <= 0 || by <= 0 || (bx + bw) >= W || (by + bh) >= H) continue;
        RegionData rd;
        rd.label = l; rd.area = a;
        rd.cx = ct.at<double>(l, 0); rd.cy = ct.at<double>(l, 1);
        rd.x = bx; rd.y = by; rd.width = bw; rd.height = bh;
        rd.color = cv::Vec3b(0, 0, 0);
        valid.push_back(rd);
    }
    std::sort(valid.begin(), valid.end(),
        [](const RegionData& a, const RegionData& b) {return a.area > b.area; });
    if ((int)valid.size() > maxReg) valid.resize(maxReg);
    for (int i = 0; i < (int)valid.size(); i++)
        valid[i].color = COLOR_PALETTE[i % COLOR_PALETTE.size()];
    regionMap = cv::Mat::zeros(cleaned.size(), CV_8UC3);
    for (const RegionData& rd : valid) {
        for (int r = 0; r < lm.rows; r++)
            for (int c = 0; c < lm.cols; c++)
                if (lm.at<int>(r, c) == rd.label)
                    regionMap.at<cv::Vec3b>(r, c) = rd.color;
        cv::rectangle(regionMap, cv::Point(rd.x, rd.y),
            cv::Point(rd.x + rd.width, rd.y + rd.height), cv::Scalar(255, 255, 255), 1);
        cv::circle(regionMap, cv::Point((int)rd.cx, (int)rd.cy), 5, cv::Scalar(255, 255, 255), -1);
        cv::putText(regionMap, "A:" + std::to_string(rd.area), cv::Point(rd.x, rd.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
    }
    return valid;
}

// TASK 4 - COMPUTE FEATURES
// Now also stores Dmin/Dmax/dmin/dmax in FeatureData for Task 8 ROI.
FeatureData computeFeatures(const cv::Mat& labelMap,
    const RegionData& rd, cv::Mat& out)
{
    FeatureData fd;
    cv::Mat mask = cv::Mat::zeros(labelMap.size(), CV_8UC1);
    for (int r = 0; r < labelMap.rows; r++)
        for (int c = 0; c < labelMap.cols; c++)
            if (labelMap.at<int>(r, c) == rd.label) mask.at<uchar>(r, c) = 255;

    cv::Moments M = cv::moments(mask, true);
    double mu20 = M.mu20, mu11 = M.mu11, mu02 = M.mu02;
    double e20 = M.nu20, e11 = M.nu11, e02 = M.nu02;

    fd.alpha = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02);
    double cA = std::cos(fd.alpha), sA = std::sin(fd.alpha);
    cv::Point2f vmaj((float)cA, (float)sA), vmin(-(float)sA, (float)cA);
    cv::Point2f cen((float)rd.cx, (float)rd.cy);

    double Dmin = 1e9, Dmax = -1e9, dmin = 1e9, dmax = -1e9;
    for (int r = 0; r < mask.rows; r++)
        for (int c = 0; c < mask.cols; c++) {
            if (!mask.at<uchar>(r, c)) continue;
            float dx = c - (float)rd.cx, dy = r - (float)rd.cy;
            double pM = dx * vmaj.x + dy * vmaj.y, pm = dx * vmin.x + dy * vmin.y;
            Dmin = std::min(Dmin, pM); Dmax = std::max(Dmax, pM);
            dmin = std::min(dmin, pm); dmax = std::max(dmax, pm);
        }

    // Store extents for Task 8
    fd.Dmin = Dmin; fd.Dmax = Dmax; fd.dmin = dmin; fd.dmax = dmax;

    double obbH = Dmax - Dmin, obbW = dmax - dmin;
    fd.obbCorners[0] = cen + (float)Dmin * vmaj + (float)dmin * vmin;
    fd.obbCorners[1] = cen + (float)Dmin * vmaj + (float)dmax * vmin;
    fd.obbCorners[2] = cen + (float)Dmax * vmaj + (float)dmax * vmin;
    fd.obbCorners[3] = cen + (float)Dmax * vmaj + (float)dmin * vmin;
    fd.axisStart = cen + (float)Dmin * vmaj;
    fd.axisEnd = cen + (float)Dmax * vmaj;

    fd.percentFilled = (obbH * obbW > 0) ? (rd.area / (obbH * obbW)) : 0.0;
    fd.hwRatio = (obbW > 0) ? (obbH / obbW) : 0.0;
    double term = std::sqrt(4.0 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02));
    double l0 = (mu20 + mu02) / 2.0 + term / 2.0, l1 = (mu20 + mu02) / 2.0 - term / 2.0;
    fd.eccentricity = (l0 > 0) ? std::sqrt(1.0 - l1 / l0) : 0.0;
    fd.huMoment1 = e20 + e02;
    fd.huMoment2 = (e20 - e02) * (e20 - e02) + 4.0 * e11 * e11;

    for (int i = 0; i < 4; i++)
        cv::line(out,
            cv::Point((int)fd.obbCorners[i].x, (int)fd.obbCorners[i].y),
            cv::Point((int)fd.obbCorners[(i + 1) % 4].x, (int)fd.obbCorners[(i + 1) % 4].y),
            cv::Scalar(255, 255, 255), 2);
    cv::line(out, cv::Point((int)fd.axisStart.x, (int)fd.axisStart.y),
        cv::Point((int)fd.axisEnd.x, (int)fd.axisEnd.y), cv::Scalar(0, 255, 255), 2);
    cv::circle(out, cv::Point((int)rd.cx, (int)rd.cy), 5, cv::Scalar(0, 0, 255), -1);
    std::string fs = "Fill:" + std::to_string(fd.percentFilled).substr(0, 5)
        + " H/W:" + std::to_string(fd.hwRatio).substr(0, 4);
    cv::putText(out, fs, cv::Point(rd.x, rd.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
    return fd;
}

// TASK 5 - TRAINING DATA I/O
const std::string TRAINING_FILE = "training_data.csv";
const std::string EMBEDDING_FILE = "embedding_db.csv";

void saveTrainingEntry(const std::string& label, const FeatureData& fd)
{
    std::ofstream f(TRAINING_FILE, std::ios::app);
    if (!f.is_open()) { std::cerr << "Cannot open training file.\n"; return; }
    f << label << "," << fd.hwRatio << "," << fd.percentFilled << ","
        << fd.eccentricity << "," << fd.huMoment1 << "," << fd.huMoment2 << "\n";
    f.close();
    std::cout << "Saved feature entry: [" << label << "]\n";
}

std::vector<TrainingEntry> loadTrainingData()
{
    std::vector<TrainingEntry> db;
    std::ifstream f(TRAINING_FILE);
    if (!f.is_open()) { std::cout << "No training file found.\n"; return db; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        TrainingEntry e;
        std::getline(ss, e.label, ',');
        std::getline(ss, tok, ','); e.hwRatio = std::stod(tok);
        std::getline(ss, tok, ','); e.percentFilled = std::stod(tok);
        std::getline(ss, tok, ','); e.eccentricity = std::stod(tok);
        std::getline(ss, tok, ','); e.huMoment1 = std::stod(tok);
        std::getline(ss, tok, ','); e.huMoment2 = std::stod(tok);
        db.push_back(e);
    }
    std::cout << "Loaded " << db.size() << " feature training entries.\n";
    return db;
}

// TASK 6 - CLASSIFICATION (scaled Euclidean nearest-neighbor)
void computeStdDevs(const std::vector<TrainingEntry>& db,
    double& sHW, double& sFill, double& sEcc, double& sH1, double& sH2)
{
    int n = (int)db.size();
    if (n < 2) { sHW = sFill = sEcc = sH1 = sH2 = 1.0; return; }
    double mHW = 0, mF = 0, mE = 0, m1 = 0, m2 = 0;
    for (const auto& e : db) { mHW += e.hwRatio; mF += e.percentFilled; mE += e.eccentricity; m1 += e.huMoment1; m2 += e.huMoment2; }
    mHW /= n; mF /= n; mE /= n; m1 /= n; m2 /= n;
    double vHW = 0, vF = 0, vE = 0, v1 = 0, v2 = 0;
    for (const auto& e : db) {
        vHW += (e.hwRatio - mHW) * (e.hwRatio - mHW);
        vF += (e.percentFilled - mF) * (e.percentFilled - mF);
        vE += (e.eccentricity - mE) * (e.eccentricity - mE);
        v1 += (e.huMoment1 - m1) * (e.huMoment1 - m1);
        v2 += (e.huMoment2 - m2) * (e.huMoment2 - m2);
    }
    sHW = std::max(0.0001, std::sqrt(vHW / n));
    sFill = std::max(0.0001, std::sqrt(vF / n));
    sEcc = std::max(0.0001, std::sqrt(vE / n));
    sH1 = std::max(0.0001, std::sqrt(v1 / n));
    sH2 = std::max(0.0001, std::sqrt(v2 / n));
}

ClassificationResult classifyRegion(const FeatureData& fd,
    const std::vector<TrainingEntry>& db,
    double sHW, double sFill, double sEcc,
    double sH1, double sH2, double unkT = 5.0)
{
    ClassificationResult r{ "unknown",std::numeric_limits<double>::max(),true };
    for (const TrainingEntry& e : db) {
        double d = std::sqrt(
            std::pow((fd.hwRatio - e.hwRatio) / sHW, 2) +
            std::pow((fd.percentFilled - e.percentFilled) / sFill, 2) +
            std::pow((fd.eccentricity - e.eccentricity) / sEcc, 2) +
            std::pow((fd.huMoment1 - e.huMoment1) / sH1, 2) +
            std::pow((fd.huMoment2 - e.huMoment2) / sH2, 2));
        if (d < r.distance) { r.distance = d; r.label = e.label; }
    }
    r.isUnknown = (r.distance > unkT);
    if (r.isUnknown) r.label = "unknown";
    return r;
}

// TASK 9 - PRE-PROCESS ROI
//
// Steps A-C as described in the task:
// A. Rotate the original frame by -alpha around the region centroid so
//    the major axis aligns with the X-axis.
// B. Extract the axis-aligned ROI using the OBB extents Dmin/Dmax/dmin/dmax.
//    After rotating by -alpha, the OBB becomes a plain rectangle.
// C. Resize the extracted ROI to 224x224 for ResNet18 input.
//
// Returns a 224x224 BGR image, or an empty Mat if extraction fails.
cv::Mat preprocessROI(const cv::Mat& frame, const RegionData& rd, const FeatureData& fd)
{
    // A: Rotate frame by -alpha around the region centroid
    cv::Point2f center((float)rd.cx, (float)rd.cy);
    double angleDeg = -fd.alpha * 180.0 / CV_PI;
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    cv::Mat rotated;
    cv::warpAffine(frame, rotated, rotMat, frame.size(),
        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // B: After rotation by -alpha, the OBB extents define an axis-aligned box.
    // cx + Dmin -> left edge,  cx + Dmax -> right edge  (along X, major axis)
    // cy + dmin -> top edge,   cy + dmax -> bottom edge (along Y, minor axis)
    int x1 = (int)std::round(rd.cx + fd.Dmin);
    int x2 = (int)std::round(rd.cx + fd.Dmax);
    int y1 = (int)std::round(rd.cy + fd.dmin);
    int y2 = (int)std::round(rd.cy + fd.dmax);

    // Clamp to image bounds
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(rotated.cols - 1, x2);
    y2 = std::min(rotated.rows - 1, y2);

    if (x2 <= x1 || y2 <= y1) return cv::Mat();  // degenerate region

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat cropped = rotated(roi).clone();

    // C: Resize to 224x224 for ResNet18
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(224, 224));
    return resized;
}

// TASK 9 - COMPUTE EMBEDDING
//
// Runs a 224x224 BGR image through a pre-loaded ResNet18 ONNX network.
// Extracts the output of the "avgpool" layer — the second-to-last layer
// before the final fully-connected classifier — giving a 512-dim vector.
// This is the image embedding used for one-shot classification.
std::vector<float> computeEmbedding(const cv::Mat& roi224, cv::dnn::Net& net)
{
    if (roi224.empty()) return {};

    // Convert BGR image to a blob:
    // - Scale pixel values from [0,255] to [0,1] (divide by 255)
    // - Apply ImageNet mean subtraction: mean = [0.485, 0.456, 0.406]
    // - Apply ImageNet std division:     std  = [0.229, 0.224, 0.225]
    // - Output size: [1, 3, 224, 224]
    cv::Mat blob = cv::dnn::blobFromImage(
        roi224,
        1.0 / 255.0,
        cv::Size(224, 224),
        cv::Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255),
        true,   // swapRB: convert BGR -> RGB
        false);

    net.setInput(blob);

    // Forward pass up to the avgpool layer.
    // In the ResNet18 ONNX from the professor, the avgpool output
    // is a [1, 512, 1, 1] tensor. We flatten it to a 512-dim vector.
    cv::Mat output = net.forward("onnx_node!resnetv22_pool1_fwd");

    // Flatten [1,512,1,1] -> vector of 512 floats
    std::vector<float> embedding(output.begin<float>(), output.end<float>());
    return embedding;
}

// TASK 9 - EMBEDDING DATABASE I/O
// Each row: label, e0, e1, e2, ..., e511
void saveEmbeddingEntry(const std::string& label, const std::vector<float>& emb)
{
    std::ofstream f(EMBEDDING_FILE, std::ios::app);
    if (!f.is_open()) { std::cerr << "Cannot open embedding file.\n"; return; }
    f << label;
    for (float v : emb) f << "," << v;
    f << "\n";
    f.close();
    std::cout << "Saved embedding for [" << label << "] ("
        << emb.size() << " dims)\n";
}

std::vector<EmbeddingEntry> loadEmbeddingDB()
{
    std::vector<EmbeddingEntry> db;
    std::ifstream f(EMBEDDING_FILE);
    if (!f.is_open()) { std::cout << "No embedding file found.\n"; return db; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        EmbeddingEntry e;
        std::getline(ss, e.label, ',');
        while (std::getline(ss, tok, ','))
            e.embedding.push_back(std::stof(tok));
        if (!e.embedding.empty()) db.push_back(e);
    }
    std::cout << "Loaded " << db.size() << " embedding entries.\n";
    return db;
}

// TASK 9 - CLASSIFY BY EMBEDDING
// Uses sum-squared difference (SSD) as the distance metric.
// SSD = sum over all dims of (query[i] - entry[i])^2
// Returns the label of the nearest neighbor embedding.
ClassificationResult classifyByEmbedding(const std::vector<float>& query,
    const std::vector<EmbeddingEntry>& db,
    double unkThresh = 50000.0)
{
    ClassificationResult r{ "unknown", std::numeric_limits<double>::max(), true };
    if (query.empty() || db.empty()) return r;

    for (const EmbeddingEntry& e : db) {
        if (e.embedding.size() != query.size()) continue;
        double ssd = 0.0;
        for (int i = 0; i < (int)query.size(); i++) {
            double diff = query[i] - e.embedding[i];
            ssd += diff * diff;
        }
        if (ssd < r.distance) { r.distance = ssd; r.label = e.label; }
    }
    r.isUnknown = (r.distance > unkThresh);
    if (r.isUnknown) r.label = "unknown";
    return r;
}

// MAIN
int main()
{
    cv::VideoCapture cap("http://10.0.0.106:8080/video", cv::CAP_FFMPEG);
    if (!cap.isOpened()) { std::cerr << "ERROR: Cannot connect.\n"; return -1; }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Load hand-crafted training data (Task 5/6)
    std::vector<TrainingEntry> trainDB = loadTrainingData();
    double sHW, sFill, sEcc, sH1, sH2;
    computeStdDevs(trainDB, sHW, sFill, sEcc, sH1, sH2);

    // Load ResNet18 ONNX model (Task 9)
    // Place resnet18.onnx in the same directory as your executable.
    cv::dnn::Net resnet;
    bool netLoaded = false;
    try {
        resnet = cv::dnn::readNetFromONNX("C:/Users/ahiva/source/repos/Project3/x64/Debug/resnet18-v2-7.onnx");
        resnet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        resnet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        netLoaded = true;
        std::cout << "ResNet18 loaded successfully.\n";
    }
    catch (...) {
        std::cerr << "WARNING: Could not load resnet18.onnx. "
            << "Task 9 embedding classification disabled.\n";
    }

    // Load embedding database (Task 9)
    std::vector<EmbeddingEntry> embDB = loadEmbeddingDB();

    // Confusion matrix (Task 7)
    ConfusionMatrix confMatrix;
    confMatrix.initLabels(trainDB);

    ConfusionMatrix embConfMatrix;
    embConfMatrix.initLabels(trainDB);

    int satThresh = 60, threshOffset = 0, saveCount = 0;

    std::cout << "\nControls:\n"
        << "  't' - save hand-crafted feature training entry\n"
        << "  'r' - save embedding training entry (Task 8)\n"
        << "  'e' - evaluate: record true label for confusion matrix\n"
        << "  'p' - print + save confusion matrix\n"
        << "  's' - save images\n"
        << "  '+'/'-' - adjust threshold\n"
        << "  'q' - quit\n\n";

    while (true) {
        cap.grab(); cap.grab();
        cv::Mat frame;
        cap.retrieve(frame);
        if (frame.empty()) { std::cerr << "Empty frame.\n"; break; }
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::Mat blurred, hsv;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
        cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> ch;
        cv::split(hsv, ch);
        int dynT = isodataThreshold(ch[2]) + threshOffset;
        dynT = std::max(10, std::min(245, dynT));

        cv::Mat binary = customThreshold(hsv, dynT, satThresh);
        cv::Mat cleaned = morphologicalCleanup(binary, 3);

        cv::Mat regionMap;
        std::vector<RegionData> regions = segmentRegions(cleaned, regionMap, 500, 5);

        cv::Mat lm, st, ct;
        cv::connectedComponentsWithStats(cleaned, lm, st, ct, 8, CV_32S);

        cv::Mat featDisp = frame.clone();
        std::vector<FeatureData> features;
        for (const RegionData& rd : regions)
            features.push_back(computeFeatures(lm, rd, featDisp));

        // Task 6: hand-crafted classification
        std::vector<ClassificationResult> hcResults;
        for (int i = 0; i < (int)features.size(); i++) {
            ClassificationResult cr = classifyRegion(
                features[i], trainDB, sHW, sFill, sEcc, sH1, sH2, 5.0);
            hcResults.push_back(cr);
            cv::Scalar col = cr.isUnknown ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::putText(featDisp, "HC:" + cr.label,
                cv::Point(regions[i].x, regions[i].y - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2);
        }

        // Task 9: embedding classification
        // Runs only if ResNet18 loaded and embedding DB has entries
        cv::Mat embDisp = frame.clone();
        std::vector<ClassificationResult> embResults;
        if (netLoaded) {
            for (int i = 0; i < (int)features.size(); i++) {
                cv::Mat roi = preprocessROI(frame, regions[i], features[i]);
                std::vector<float> emb = computeEmbedding(roi, resnet);
                ClassificationResult er = classifyByEmbedding(emb, embDB, 50000.0);
                embResults.push_back(er);

                // Draw OBB and axis on embedding display too
                computeFeatures(lm, regions[i], embDisp);

                cv::Scalar col = er.isUnknown ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::putText(embDisp, "EM:" + er.label,
                    cv::Point(regions[i].x, regions[i].y - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2);
                std::string distStr = "d:" + std::to_string((int)er.distance);
                cv::putText(embDisp, distStr,
                    cv::Point(regions[i].x, regions[i].y - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
            }
        }
        else {
            cv::putText(embDisp, "ResNet18 not loaded",
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 0, 255), 2);
        }

        // Build 2x3 display grid
        cv::Mat bCol, clCol;
        cv::cvtColor(binary, bCol, cv::COLOR_GRAY2BGR);
        cv::cvtColor(cleaned, clCol, cv::COLOR_GRAY2BGR);

        auto label = [](cv::Mat& m, const std::string& t, cv::Scalar c = { 0,255,0 }) {
            cv::putText(m, t, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 2);
            };
        label(frame, "Original");
        label(bCol, "Thresholded");
        label(clCol, "Cleaned");
        label(regionMap, "Regions", cv::Scalar(255, 255, 255));
        label(featDisp, "HC Class", cv::Scalar(0, 255, 0));
        label(embDisp, "Embed Class", cv::Scalar(0, 255, 255));

        cv::putText(featDisp, "DB:" + std::to_string(trainDB.size()),
            cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
        cv::putText(embDisp, "Emb:" + std::to_string(embDB.size()),
            cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);

        cv::Mat top, bot, display;
        cv::hconcat(std::vector<cv::Mat>{frame, bCol, clCol}, top);
        cv::hconcat(std::vector<cv::Mat>{regionMap, featDisp, embDisp}, bot);
        cv::vconcat(top, bot, display);

        cv::putText(display, "Thresh:" + std::to_string(dynT) + " Sat:" + std::to_string(satThresh),
            cv::Point(10, display.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::imshow("Object Recognition Pipeline", display);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;

        // Task 5: save hand-crafted training entry
        if (key == 't') {
            if (regions.empty()) { std::cout << "No region detected.\n"; }
            else {
                std::cout << "Enter label: "; std::string lbl;
                std::getline(std::cin, lbl);
                if (!lbl.empty()) {
                    saveTrainingEntry(lbl, features[0]);
                    TrainingEntry e;
                    e.label = lbl; e.hwRatio = features[0].hwRatio;
                    e.percentFilled = features[0].percentFilled;
                    e.eccentricity = features[0].eccentricity;
                    e.huMoment1 = features[0].huMoment1;
                    e.huMoment2 = features[0].huMoment2;
                    trainDB.push_back(e);
                    computeStdDevs(trainDB, sHW, sFill, sEcc, sH1, sH2);
                    confMatrix.initLabels(trainDB);
                }
            }
        }

        // Task 9: save embedding training entry
        if (key == 'r') {
            if (!netLoaded) { std::cout << "ResNet18 not loaded.\n"; }
            else if (regions.empty()) { std::cout << "No region detected.\n"; }
            else {
                std::cout << "Enter label for embedding: "; std::string lbl;
                std::getline(std::cin, lbl);
                if (!lbl.empty()) {
                    cv::Mat roi = preprocessROI(frame, regions[0], features[0]);
                    std::vector<float> emb = computeEmbedding(roi, resnet);
                    if (!emb.empty()) {
                        saveEmbeddingEntry(lbl, emb);
                        EmbeddingEntry ee; ee.label = lbl; ee.embedding = emb;
                        embDB.push_back(ee);
                    }
                }
            }
        }

        // Task 7: evaluation
        if (key == 'e') {
            if (regions.empty()) { std::cout << "No region detected.\n"; }
            else {
                std::string hcPred = hcResults.empty() ? "none" : hcResults[0].label;
                std::string emPred = embResults.empty() ? "none" : embResults[0].label;
                std::cout << "HC predicted: [" << hcPred << "]  Emb predicted: [" << emPred << "]\n";
                std::cout << "Enter TRUE label: "; std::string tl;
                std::getline(std::cin, tl);
                if (!tl.empty()) {
                    confMatrix.record(tl, hcPred);
                    if (!embResults.empty())
                        embConfMatrix.record(tl, emPred);
                    bool ok = (tl == hcPred);
                    std::cout << (ok ? "CORRECT" : "INCORRECT")
                        << " | True:" << tl << " | HC:" << hcPred << " | Emb:" << emPred
                        << " | Acc:" << std::fixed << std::setprecision(1)
                        << (100.0 * confMatrix.correct / confMatrix.total) << "%\n";
                }
            }
        }

        if (key == 'p') {
            std::cout << "\n--- HAND-CRAFTED FEATURES ---\n";
            confMatrix.print();
            confMatrix.saveToFile("confusion_matrix_HC.csv");

            std::cout << "\n--- RESNET18 EMBEDDING ---\n";
            embConfMatrix.print();
            embConfMatrix.saveToFile("confusion_matrix_Emb.csv");
        }

        if (key == 's') {
            cv::imwrite("orig_" + std::to_string(saveCount) + ".png", frame);
            cv::imwrite("cleaned_" + std::to_string(saveCount) + ".png", cleaned);
            cv::imwrite("regions_" + std::to_string(saveCount) + ".png", regionMap);
            cv::imwrite("hc_class_" + std::to_string(saveCount) + ".png", featDisp);
            cv::imwrite("em_class_" + std::to_string(saveCount) + ".png", embDisp);
            std::cout << "Saved all images for sample " << saveCount << "\n";
            saveCount++;
        }

        if (key == '+' || key == '=') threshOffset += 5;
        if (key == '-' || key == '_') threshOffset -= 5;
    }

    if (confMatrix.total > 0) {
        std::cout << "\n--- FINAL: HAND-CRAFTED FEATURES ---\n";
        confMatrix.print();
        confMatrix.saveToFile("confusion_matrix_HC.csv");

        std::cout << "\n--- FINAL: RESNET18 EMBEDDING ---\n";
        embConfMatrix.print();
        embConfMatrix.saveToFile("confusion_matrix_Emb.csv");
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}