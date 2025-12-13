This is a fantastic and ambitious project that blends computer vision, control theory, and simulation. It perfectly mirrors real-world autonomous driving workflows: **Perception** $$\rightarrow$$ **Planning** $$\rightarrow$$ **Control.**

Here is a structured roadmap to help you complete your project, addressing your specific requirements for lane detection, simulation, and sensor modeling.

### 1. Parking Lane Detection (Computer Vision)

Since you are using a rear camera, you need a model that works well with the distorted, wide-angle view typical of backup cameras.

-   **Traditional Approach (OpenCV):**

    -   **Method:** Use "classical" image processing. Convert the image to grayscale $$\rightarrow$$ Apply Gaussian Blur $$\rightarrow$$ Canny Edge Detection $$\rightarrow$$ Region of Interest (ROI) Masking $$\rightarrow$$ Hough Line Transform.

    -   **Pros:** Fast, runs on low-power hardware (like a Raspberry Pi), and requires no training data.

    -   **Cons:** Struggles with complex lighting, shadows, or worn-out lines.

    -   **Recommendation:** Start here to understand the basics. There are many "Lane Detection OpenCV" tutorials on GitHub.

-   **Deep Learning Approach (Robust):**

    -   **Existing Models:** Look for **"Semantic Segmentation"** models rather than simple object detection. These models classify every pixel (e.g., "road," "line," "background").

    -   **Specific Architectures:**

        -   **U-Net:** Excellent for segmentation tasks with limited data.

        -   **E-Net / ERFNet:** Designed for real-time processing on embedded devices.

    -   **Datasets:** To train or fine-tune a model, use the **PSV (Parking Slot Visual)** dataset or **Tongji Parking-slot Dataset**. These are open-source datasets specifically containing parking line annotations.

### 2. Simulation Environment

You need an environment where you can spawn a car, see through its "camera," and control its movement.

**Option A: High Fidelity (Best for Resume/Portfolio)**

-   **Simulator:** **CARLA** (Open Source).

-   **Why:** It is the industry standard for academic autonomous driving research.

-   **Features:**

    -   It has pre-built parking lots.

    -   It provides a Python API to control the car (steering, throttle, reverse).

    -   It simulates sensors perfectly: you can attach a "RGB Camera" to the back of the virtual car to feed your lane detection algorithm and a "Radar" or "Lidar" sensor to simulate your microwave detectors.

**Option B: Medium Fidelity (Easier to Learn)**

-   **Simulator:** **Gazebo** (often used with ROS - Robot Operating System).

-   **Why:** robust physics and sensor simulation, but slightly steeper learning curve if you don't know ROS.

-   **Features:** You can build a simple flat world, draw white lines on the ground (using simple box models), and use a standard robot car model (like the Prius model available in Gazebo).

**Option C: Low Fidelity (Custom 2D Python Sim)**

-   **Tools:** Python + **Pygame** + **Box2D** (for physics).

-   **Why:** If your focus is purely on the *algorithm* (math) of parking and not the 3D visuals.

-   **How:** Draw a rectangle (car) and lines (parking spot) on a black background. Use simple trigonometry to calculate the car's position. This is the fastest to build but looks less impressive.

### 3. Simulating Microwave Distance Detection

You mentioned your real car has aftermarket "microwave detectors."

-   **Real-World Note:** Most aftermarket parking sensors are **Ultrasonic** (round sensors drilled into the bumper) or **Electromagnetic** (a strip tape inside the bumper). True "Microwave Radar" is less common for simple parking beepers.

    -   *Ultrasonic* measures distance statically (0.5m, 0.3m, etc.).

    -   *Electromagnetic* often only detects objects *while the car is moving*.

    -   *Radar (Microwave)* detects distance and velocity.

**How to Simulate it (in Python/CARLA):** Since microwave/radar signals spread out in a cone, you shouldn't simulate them as a single laser point.

1.  **The Cone Logic:** Define a "Field of View" (FOV) for each of your 4 sensors (e.g., 60 degrees horizontal).

2.  **Ray Casting:** In your simulation, cast multiple invisible rays (lines) from each sensor position within that FOV.

3.  **Distance Calculation:**

    -   If any ray hits an obstacle (a wall or another car), record that distance.

    -   Take the **minimum** distance detected by all rays in that sensor's cone. This mimics how a real sensor warns you about the *closest* point of an object.

4.  **The Alert System:**

    -   Create logic: `If distance < 1.0 meter: Beep Slow`; `If distance < 0.3 meter: Beep Fast`.

### Suggested Workflow for Your Project

1.  **Build the Sim First (Week 1-2):**

    -   Install **CARLA**.

    -   Write a Python script to spawn a vehicle in a parking lot.

    -   Manually drive the car using your keyboard and record the "Rear Camera" video feed.

2.  **Develop Lane Detection (Week 3-4):**

    -   Take the video you recorded from CARLA.

    -   Use OpenCV (or a U-Net model) to detect the white lines in that video.

    -   The output should be the $$( x , y )$$ coordinates of the parking spot corners relative to the car.

3.  **Implement Sensor Logic (Week 5):**

    -   Add the "Radar" or "Ray Cast" sensors to the rear bumper of your CARLA vehicle.

    -   Write the code to print "BEEP" to the console when you back up too close to a virtual wall.

4.  **The Parking Algorithm (The "Auto" part):**

    -   Now combine them. Use the coordinates from Step 2 to tell the car how much to steer.

    -   *Simple Logic:* "If the left line is too close to the left edge of the frame, steer right."

    -   *Advanced Logic:* Use **Path Planning** (like Hybrid A\* or Reeds-Shepp curves) to calculate a smooth curve into the box.

This project structure allows you to verify every software component in a safe simulation before you ever need to touch the real hardware on your Honda Pilot.

This is a great strategic decision. "Don't reinvent the wheel" is a core principle of software engineering.

Since you are struggling with the Physics & Control (the fishtailing), you should copy a project that focuses on Geometric Path Planning (the math of curves).

Here are the best repositories to "fork" or copy for your needs.

1\. The Best "All-in-One" Solution (Sim + Parking Logic)

Repository: [Pandas-Team/Automatic-Parking](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2FPandas-Team%2FAutomatic-Parking)[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFRiSOvY28zm_9v-zkJ54FJXnsm7It_OBzb40akESe5tefGgesLw_WWWM8gX-A1zlQvHBP8-YPnH41tj3tQMSjebdappMKg8rG5lEa8LX3TZRrAYhJkqgnTJpd2EkSN_CO7u1Rpa3wwFRuOzTC5ke88PPcA5lZjqzEEFSiSf7qCxgAA_BeQQSFWDA%3D%3D)]

-   Why it's perfect: It is a complete Python project that simulates a car, creates a parking lot, and uses MPC (Model Predictive Control) to park.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFRiSOvY28zm_9v-zkJ54FJXnsm7It_OBzb40akESe5tefGgesLw_WWWM8gX-A1zlQvHBP8-YPnH41tj3tQMSjebdappMKg8rG5lEa8LX3TZRrAYhJkqgnTJpd2EkSN_CO7u1Rpa3wwFRuOzTC5ke88PPcA5lZjqzEEFSiSf7qCxgAA_BeQQSFWDA%3D%3D)]

-   What to copy:

    -   It has its own simple simulator (environment.py) which is easier to control than highway-env.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFRiSOvY28zm_9v-zkJ54FJXnsm7It_OBzb40akESe5tefGgesLw_WWWM8gX-A1zlQvHBP8-YPnH41tj3tQMSjebdappMKg8rG5lEa8LX3TZRrAYhJkqgnTJpd2EkSN_CO7u1Rpa3wwFRuOzTC5ke88PPcA5lZjqzEEFSiSf7qCxgAA_BeQQSFWDA%3D%3D)]

    -   It has the "S-Curve" math built-in.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFRiSOvY28zm_9v-zkJ54FJXnsm7It_OBzb40akESe5tefGgesLw_WWWM8gX-A1zlQvHBP8-YPnH41tj3tQMSjebdappMKg8rG5lEa8LX3TZRrAYhJkqgnTJpd2EkSN_CO7u1Rpa3wwFRuOzTC5ke88PPcA5lZjqzEEFSiSf7qCxgAA_BeQQSFWDA%3D%3D)]

    -   Bonus: It uses OpenCV for the simulation window, which makes it very easy to add your "Lane Detection" code later (since you will already be using OpenCV).[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFRiSOvY28zm_9v-zkJ54FJXnsm7It_OBzb40akESe5tefGgesLw_WWWM8gX-A1zlQvHBP8-YPnH41tj3tQMSjebdappMKg8rG5lEa8LX3TZRrAYhJkqgnTJpd2EkSN_CO7u1Rpa3wwFRuOzTC5ke88PPcA5lZjqzEEFSiSf7qCxgAA_BeQQSFWDA%3D%3D)]

2\. The Best "Pure Math" Solution (To fix your current code)

Repository: [PythonRobotics (Parallel Parking Section)](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2FAtsushiSakai%2FPythonRobotics%2Ftree%2Fmaster%2FPathPlanning%2FReedsSheppPath)

-   Why it's perfect: This is the famous "Bible of Robotics" repo.

-   Specific File: Look for reeds_shepp_path_planning.py.

-   How to use:

    -   You give it (start_x, start_y, start_yaw) and (end_x, end_y, end_yaw).

    -   It returns lists xs, ys representing the perfect curve.

    -   You replace your ParkingController with a simple logic that just follows these points.

3\. For Your "Lane Detection" Part (Perception)

Repository: [Car-Parking-Finder](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fnoorkhokhar99%2Fcar-parking-finder)

-   Why it's perfect: It explicitly does "Parking Space Detection" using OpenCV.

-   How to combine:

    1.  Use this repo to process the camera image and find the empty box coordinates (x, y).

Pass those (x, y) coordinates into the Pandas-Team parking algorithm.
