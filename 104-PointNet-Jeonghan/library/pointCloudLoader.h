#ifndef POINT_CLOUD_LOADER_H
#define POINT_CLOUD_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>

/**
 * Point Cloud Data Loader
 * 
 * Utilities for loading point cloud data from various file formats:
 * - .txt: Simple text format (x y z per line)
 * - .ply: Polygon File Format (ASCII only for now)
 * - .xyz: XYZ format
 */

namespace pointcloud {

/**
 * Load point cloud from text file (.txt, .xyz)
 * Format: Each line contains "x y z" (space or comma separated)
 * 
 * @param filename Path to the point cloud file
 * @return Vector of points [N, 3] flattened as [x1, y1, z1, x2, y2, z2, ...]
 */
inline std::vector<float> loadFromTxt(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::vector<float> points;
    std::string line;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Replace commas with spaces for flexible parsing
        std::replace(line.begin(), line.end(), ',', ' ');
        
        std::istringstream iss(line);
        float x, y, z;
        
        if (iss >> x >> y >> z) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
        }
    }
    
    file.close();
    
    if (points.empty()) {
        throw std::runtime_error("No valid points found in file: " + filename);
    }
    
    return points;
}

/**
 * Load point cloud from PLY file (ASCII format only)
 * 
 * @param filename Path to the .ply file
 * @return Vector of points [N, 3] flattened
 */
inline std::vector<float> loadFromPly(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::vector<float> points;
    std::string line;
    bool in_header = true;
    int num_vertices = 0;
    
    // Parse header
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string elem, vertex;
            iss >> elem >> vertex >> num_vertices;
        }
        
        if (line.find("end_header") != std::string::npos) {
            in_header = false;
            break;
        }
    }
    
    if (num_vertices == 0) {
        throw std::runtime_error("Invalid PLY file (no vertex count): " + filename);
    }
    
    // Read vertex data
    points.reserve(num_vertices * 3);
    
    for (int i = 0; i < num_vertices && std::getline(file, line); ++i) {
        std::istringstream iss(line);
        float x, y, z;
        
        if (iss >> x >> y >> z) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
        }
    }
    
    file.close();
    
    if (points.empty()) {
        throw std::runtime_error("No valid points found in PLY file: " + filename);
    }
    
    return points;
}

/**
 * Load point cloud from OFF file (Object File Format)
 * Format:
 *   OFF
 *   num_vertices num_faces num_edges
 *   x y z (for each vertex)
 *   ... (face data, which we ignore)
 *
 * Also handles combined format: OFF9293 10182 0 (header on same line as counts)
 *
 * @param filename Path to the .off file
 * @return Vector of points [N, 3] flattened
 */
inline std::vector<float> loadFromOff(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<float> points;
    std::string line;

    // Read first line (should be "OFF" or "OFFnnn nnn nnn")
    std::getline(file, line);
    if (line.find("OFF") == std::string::npos) {
        throw std::runtime_error("Invalid OFF file (missing header): " + filename);
    }

    // Read counts: num_vertices num_faces num_edges
    int num_vertices = 0, num_faces = 0, num_edges = 0;

    // Check if counts are on the same line as OFF (e.g., "OFF9293 10182 0")
    if (line.length() > 3 && line.substr(0, 3) == "OFF") {
        std::string counts_part = line.substr(3);  // Remove "OFF" prefix
        std::istringstream count_iss(counts_part);
        count_iss >> num_vertices >> num_faces >> num_edges;
    }

    // If counts not found on first line, read next line
    if (num_vertices == 0) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Invalid OFF file (no counts): " + filename);
        }
        std::istringstream count_iss(line);
        count_iss >> num_vertices >> num_faces >> num_edges;
    }

    if (num_vertices <= 0) {
        throw std::runtime_error("Invalid OFF file (zero vertices): " + filename);
    }

    // Read vertex data
    points.reserve(num_vertices * 3);

    for (int i = 0; i < num_vertices && std::getline(file, line); ++i) {
        std::istringstream iss(line);
        float x, y, z;

        if (iss >> x >> y >> z) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
        }
    }

    file.close();

    if (points.empty()) {
        throw std::runtime_error("No valid points found in OFF file: " + filename);
    }

    return points;
}

/**
 * Auto-detect format and load point cloud
 * 
 * @param filename Path to the point cloud file
 * @return Vector of points [N, 3] flattened
 */
inline std::vector<float> load(const std::string& filename) {
    // Get file extension
    size_t dot_pos = filename.rfind('.');
    if (dot_pos == std::string::npos) {
        throw std::runtime_error("Cannot determine file format (no extension): " + filename);
    }
    
    std::string ext = filename.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "ply") {
        return loadFromPly(filename);
    } else if (ext == "off") {
        return loadFromOff(filename);
    } else if (ext == "txt" || ext == "xyz") {
        return loadFromTxt(filename);
    } else {
        throw std::runtime_error("Unsupported file format: " + ext);
    }
}

/**
 * Get number of points from loaded data
 */
inline size_t getNumPoints(const std::vector<float>& points) {
    return points.size() / 3;
}

/**
 * Normalize point cloud to unit sphere
 * Centers the point cloud at origin and scales to fit in unit sphere
 * 
 * @param points Input/output point cloud data (modified in-place)
 */
inline void normalize(std::vector<float>& points) {
    if (points.empty()) return;
    
    size_t num_points = points.size() / 3;
    
    // Compute centroid
    float cx = 0, cy = 0, cz = 0;
    for (size_t i = 0; i < num_points; ++i) {
        cx += points[i * 3 + 0];
        cy += points[i * 3 + 1];
        cz += points[i * 3 + 2];
    }
    cx /= num_points;
    cy /= num_points;
    cz /= num_points;
    
    // Center at origin
    for (size_t i = 0; i < num_points; ++i) {
        points[i * 3 + 0] -= cx;
        points[i * 3 + 1] -= cy;
        points[i * 3 + 2] -= cz;
    }
    
    // Find max distance from origin
    float max_dist = 0;
    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * 3 + 0];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];
        float dist = std::sqrt(x*x + y*y + z*z);
        max_dist = std::max(max_dist, dist);
    }
    
    // Scale to unit sphere
    if (max_dist > 0) {
        for (size_t i = 0; i < points.size(); ++i) {
            points[i] /= max_dist;
        }
    }
}

/**
 * Print point cloud statistics
 */
inline void printStats(const std::vector<float>& points) {
    if (points.empty()) {
        std::cout << "Empty point cloud" << std::endl;
        return;
    }
    
    size_t num_points = points.size() / 3;
    
    // Compute bounds
    float min_x = points[0], max_x = points[0];
    float min_y = points[1], max_y = points[1];
    float min_z = points[2], max_z = points[2];
    
    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * 3 + 0];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];
        
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
    }
    
    std::cout << "Point Cloud Statistics:" << std::endl;
    std::cout << "  Number of points: " << num_points << std::endl;
    std::cout << "  X range: [" << min_x << ", " << max_x << "]" << std::endl;
    std::cout << "  Y range: [" << min_y << ", " << max_y << "]" << std::endl;
    std::cout << "  Z range: [" << min_z << ", " << max_z << "]" << std::endl;
}

/**
 * Resample point cloud to fixed number of points
 * Uses uniform sampling - supports both downsampling and upsampling (with replacement)
 *
 * @param points Input point cloud (xyz...)
 * @param num_samples Target number of points
 * @return Resampled point cloud
 */
inline std::vector<float> resample(const std::vector<float>& points, size_t num_samples) {
    size_t num_points = points.size() / 3;

    if (num_points == 0) return {};
    if (num_points == num_samples) return points;  // Already correct size

    std::vector<float> result;
    result.reserve(num_samples * 3);

    // Uniform sampling (works for both up and down sampling)
    for (size_t i = 0; i < num_samples; ++i) {
        size_t idx = (i * num_points) / num_samples;
        // Clamp index to valid range (safety for edge cases)
        if (idx >= num_points) idx = num_points - 1;
        result.push_back(points[idx * 3 + 0]);
        result.push_back(points[idx * 3 + 1]);
        result.push_back(points[idx * 3 + 2]);
    }

    return result;
}

/**
 * ModelNet40 Dataset Utilities
 */
namespace modelnet40 {

inline const std::vector<std::string> CLASSES = {
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox"
};

/**
 * Get all .off files for a specific class
 */
inline std::vector<std::string> getFilesForClass(
    const std::string& dataset_path,
    const std::string& class_name,
    const std::string& split = "test"  // "train" or "test"
) {
    std::vector<std::string> files;
    std::string class_path = dataset_path + "/" + class_name + "/" + split;
    
    // Use filesystem API (C++17)
    if (!std::filesystem::exists(class_path)) {
        std::cerr << "Warning: Path not found: " << class_path << std::endl;
        return files;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(class_path)) {
        if (entry.path().extension() == ".off") {
            files.push_back(entry.path().string());
        }
    }
    
    return files;
}

/**
 * Get a random sample from ModelNet40
 */
inline std::pair<std::string, std::string> getRandomSample(
    const std::string& dataset_path,
    const std::string& split = "test"
) {
    // Pick random class
    int class_idx = rand() % CLASSES.size();
    std::string class_name = CLASSES[class_idx];
    
    // Get files for this class
    auto files = getFilesForClass(dataset_path, class_name, split);
    
    if (files.empty()) {
        return {"", ""};
    }
    
    // Pick random file
    std::string file_path = files[rand() % files.size()];
    
    return {class_name, file_path};
}

/**
 * Load and prepare ModelNet40 sample for PointNet
 * Retries up to max_retries times if loading fails
 * 
 * @param dataset_path Path to ModelNet40 root directory
 * @param num_points Number of points to sample
 * @param split "train" or "test"
 * @param max_retries Maximum number of retry attempts
 * @return Tuple of (class_name, class_index, point_cloud)
 */
inline std::tuple<std::string, int, std::vector<float>> loadSample(
    const std::string& dataset_path,
    size_t num_points = 1024,
    const std::string& split = "test",
    int max_retries = 10
) {
    for (int retry = 0; retry < max_retries; ++retry) {
        auto [class_name, file_path] = getRandomSample(dataset_path, split);
        
        if (file_path.empty()) {
            std::cerr << "Failed to find ModelNet40 sample" << std::endl;
            continue;
        }
        
        // Find class index
        int class_idx = -1;
        for (size_t i = 0; i < CLASSES.size(); ++i) {
            if (CLASSES[i] == class_name) {
                class_idx = i;
                break;
            }
        }
        
        try {
            // Load point cloud
            std::vector<float> points = loadFromOff(file_path);
            
            if (points.empty()) {
                std::cerr << "Warning: Empty points from " << file_path << ", retrying..." << std::endl;
                continue;
            }
            
            // Normalize to unit sphere
            normalize(points);
            
            // Resample to target size
            points = resample(points, num_points);
            
            std::cout << "Loaded: " << class_name << " (" << class_idx << ") from " 
                      << file_path.substr(file_path.find_last_of('/') + 1) << std::endl;
            std::cout << "Points: " << points.size() / 3 << std::endl;
            
            return {class_name, class_idx, points};
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load " << file_path << ": " << e.what() << std::endl;
            if (retry < max_retries - 1) {
                std::cerr << "Retrying with different sample..." << std::endl;
            }
        }
    }
    
    std::cerr << "Error: Failed to load valid sample after " << max_retries << " attempts" << std::endl;
    return {"", -1, {}};
}

} // namespace modelnet40

} // namespace pointcloud

#endif // POINT_CLOUD_LOADER_H
