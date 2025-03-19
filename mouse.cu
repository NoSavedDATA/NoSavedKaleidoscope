#include <X11/Xlib.h>
#include <iostream>
#include <thread>  // For sleep
#include <chrono>  // For time duration

int main() {
    // Open the display (X server)
    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        std::cerr << "Unable to open X display\n";
        return 1;
    }

    // Get the root window, which represents the entire screen
    Window root = DefaultRootWindow(display);
    
    // Variables to store mouse coordinates and window information
    int x, y, rootX, rootY;
    unsigned int mask;
    Window rootWin, childWin;

    // Capture the mouse position anywhere on the screen
    while (true) {
        // Query the pointer position on the root window
        XQueryPointer(display, root, &rootWin, &childWin, &rootX, &rootY, &x, &y, &mask);
        std::cout << "Mouse Position: " << x << ", " << y << std::endl;

        // Sleep to avoid excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Sleep for 100 ms
    }

    // Close the display connection
    XCloseDisplay(display);
    return 0;
}
