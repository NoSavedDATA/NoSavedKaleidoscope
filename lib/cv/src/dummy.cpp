

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../../../src/nsk_cpp.h"


struct Dot {
    cv::Point2f pos;
    cv::Point2f velocity;
    cv::Scalar color;
};

// Structure for a moving rectangle
struct RectObj {
    cv::Rect2f rect;
    cv::Point2f velocity;
    cv::Scalar color;
};

extern "C" void renderFrame(const std::vector<Dot>& dots, const std::vector<RectObj>& rects, int width = 640, int height = 480) {
    // Create a blank image
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(0,0,0));

    // Draw dots
    for (const auto& dot : dots) {
        cv::circle(frame, dot.pos, 5, dot.color, -1);
    }

    // Draw rectangles
    for (const auto& r : rects) {
        cv::rectangle(frame, r.rect, r.color, -1);
    }

    // Show the frame
    cv::imshow("Game", frame);
    cv::waitKey(1); // Small delay to update the window
}

// Example update function
extern "C" void updateObjects(std::vector<Dot>& dots, std::vector<RectObj>& rects, int width = 640, int height = 480) {
    for (auto& dot : dots) {
        dot.pos += dot.velocity;
        if (dot.pos.x < 0 || dot.pos.x >= width) dot.velocity.x *= -1;
        if (dot.pos.y < 0 || dot.pos.y >= height) dot.velocity.y *= -1;
    }
    for (auto& r : rects) {
        r.rect.x += r.velocity.x;
        r.rect.y += r.velocity.y;
        if (r.rect.x < 0 || r.rect.x + r.rect.width >= width) r.velocity.x *= -1;
        if (r.rect.y < 0 || r.rect.y + r.rect.height >= height) r.velocity.y *= -1;
    }
}



extern "C" float cv__dummy(Scope_Struct *scope_struct) {
    std::vector<Dot> dots = { {{100,100},{2,1},{0,0,255}}, {{200,150},{-1,2},{0,255,0}} };
    std::vector<RectObj> rects = { {{300,200,50,30},{1,-1},{255,0,0}} };

    std::cout << "Start rendering" << ".\n";

    while (true) {
        updateObjects(dots, rects);
        renderFrame(dots, rects);
    }

    return 0;
}



