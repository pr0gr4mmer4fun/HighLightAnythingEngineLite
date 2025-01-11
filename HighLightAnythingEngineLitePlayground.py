import cv2
import numpy as np
import time
from collections import deque
import cv2

print("You are using OpenCV_version: " + cv2.__version__)


class HighlightTracker:
    def __init__(self):
        # Background edge detection
        self.background_edges = None
        self.edge_buffer = deque(maxlen=5)

        # Shadow stability parameters
        self.shadow_threshold = 20
        self.min_object_size = (5, 5)
        self.min_contour_area = 20
        self.min_edge_displacement = 20

        self.highlighted_regions = []
        self.highlight_timers = {}
        self.highlight_duration = 4

        # Slash selection variables
        self.slashing = False
        self.slash_start = None
        self.slash_end = None
        self.temp_rect_timer = None
        self.temp_rect_duration = 0.1
        self.temp_rectangle = None

        # Tracking variables
        self.tracked_objects = []
        self.tracked_regions = []

        # Edge detection and smoothing parameters
        self.enable_edge_smoothing = True
        self.blur_kernel_size = (5, 5)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 110
        self.morph_kernel = np.ones((3, 3), np.uint8)
        self.morph_iterations = 1

        # Temporal smoothing parameters
        self.prev_edges = None
        self.edge_smooth_factor = 0.95

        self.base_padding_factor = 0.1

        cv2.namedWindow("HighLightAnythingEngine")
        cv2.setMouseCallback("HighLightAnythingEngine", self.click_event)

    def compute_background_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        # Apply morphological operations for stability
        edges = cv2.dilate(edges, self.morph_kernel, iterations=1)
        edges = cv2.erode(edges, self.morph_kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations)

        return edges

    def get_box_from_slash(self, start, end):
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])

        base_padding = max(min(dx if dx > dy else dy, 40), 10)
        base_padding = int(base_padding * self.base_padding_factor)

        is_horizontal = (dx > dy)

        if is_horizontal:
            padding_x = max(base_padding * 0.2, 5)
            padding_y = max(base_padding, 5)
        else:
            padding_x = max(base_padding, 5)
            padding_y = max(base_padding * 0.2, 5)

        x1 = min(start[0], end[0]) - padding_x
        y1 = min(start[1], end[1]) - padding_y
        x2 = max(start[0], end[0]) + padding_x
        y2 = max(start[1], end[1]) + padding_y

        return (int(x1), int(y1)), (int(x2), int(y2))

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.slashing = True
            self.slash_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.slashing:
            self.slash_end = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and self.slashing:
            self.slashing = False
            if self.slash_start and self.slash_end:
                p1, p2 = self.get_box_from_slash(self.slash_start, self.slash_end)
                self.temp_rectangle = (p1, p2)
                self.temp_rect_timer = time.time()
                self.process_selection(p1, p2)
                self.slash_start = None
                self.slash_end = None

    def create_tracker(self):
        try:
            if hasattr(cv2, 'legacy'):
                return cv2.legacy.TrackerKCF_create()
            else:
                return cv2.TrackerKCF_create()
        except AttributeError:
            print("TrackerKCF is not available in your OpenCV installation.")
            return None

    def process_selection(self, p1, p2):
        if not hasattr(self, 'current_frame') or self.background_edges is None:
            return

        # Get coordinates
        x1 = max(0, min(p1[0], p2[0]))
        y1 = max(0, min(p1[1], p2[1]))
        x2 = min(self.current_frame.shape[1], max(p1[0], p2[0]))
        y2 = min(self.current_frame.shape[0], max(p1[1], p2[1]))

        if (x2 - x1) < self.min_object_size[0] or (y2 - y1) < self.min_object_size[1]:
            return

        # Get the pre-computed edges for this region
        edge_roi = self.background_edges[y1:y2, x1:x2]

        # Filter small contours
        contours, _ = cv2.findContours(edge_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = np.zeros_like(edge_roi)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                cv2.drawContours(filtered_edges, [cnt], -1, 255, -1)

        # Create mask from filtered edges
        mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = filtered_edges

        # Initialize tracker
        frame_bbox = (x1, y1, x2 - x1, y2 - y1)
        tracker = self.create_tracker()
        if tracker is not None:
            success = tracker.init(self.current_frame, frame_bbox)
            if success:
                self.tracked_objects.append({
                    'tracker': tracker,
                    'initial_bbox': frame_bbox,
                    'start_time': time.time()
                })
                self.tracked_regions.append(mask)

                # Analysis
                roi = self.current_frame[y1:y2, x1:x2]
                self.analyze_roi_with_llm(roi)

    def analyze_roi_with_llm(self, roi):
        roi_path = "temp_roi.jpg"
        cv2.imwrite(roi_path, roi)
        print("\nAnalyzing selected region...")
        print(f"Selected region size: {roi.shape}")
        print(f"Average color (BGR): {np.mean(roi, axis=(0, 1))}")
        print(f"Region contains {np.sum(cv2.Canny(roi, 100, 200) > 0)} edge pixels")

        print("\nRandom BS Placeholder Analysis:")
        print(" - The fractal adjacency matrix indicates a 14% surge in hyper-pixel variance.")
        print(" - Sub-pixel harmonics imply a polychromatic locus of interest.")
        print(" - The transient wavelet transformations suggest a near-critical threshold.")
        print(" - Convergent entropic distribution approximates an isometric morphological pivot.")

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
                message = [
                    "CAMERA DISCONNECTED!",
                    "RETRYING CONNECTION...",
                    f"OpenCV_ver: {cv2.__version__}"
                ]
                x, y_start = 50, 200
                line_height = 40

                for i, line in enumerate(message):
                    y = y_start + i * line_height
                    cv2.putText(black_screen, line, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("HighLightAnythingEngine", black_screen)
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            self.current_frame = frame.copy()

            # Compute edges for entire frame
            edges = self.compute_background_edges(frame)
            self.edge_buffer.append(edges)

            if len(self.edge_buffer) == self.edge_buffer.maxlen:
                self.background_edges = np.mean(self.edge_buffer, axis=0).astype(np.uint8)
            else:
                self.background_edges = edges

            edge_mask = np.zeros_like(frame)
            display_frame = frame.copy()

            objects_to_remove = []
            current_time = time.time()

            for idx, tracked_obj in enumerate(self.tracked_objects):
                success, bbox = tracked_obj['tracker'].update(frame)

                if success and current_time - tracked_obj['start_time'] <= self.highlight_duration:
                    x, y, w, h = [int(v) for v in bbox]

                    # Get edges from pre-computed background edges
                    if 0 <= y < self.background_edges.shape[0] and 0 <= x < self.background_edges.shape[1]:
                        edge_roi = self.background_edges[y:y + h, x:x + w]
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask[y:y + h, x:x + w] = edge_roi
                        self.tracked_regions[idx] = mask

                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(edge_mask, contours, -1, (0, 242, 255), 1)
                else:
                    objects_to_remove.append(idx)

            for idx in sorted(objects_to_remove, reverse=True):
                self.tracked_objects.pop(idx)
                self.tracked_regions.pop(idx)

            if self.slashing and self.slash_start and self.slash_end:
                cv2.line(display_frame, self.slash_start, self.slash_end, (0, 255, 0), 1)

            if (self.temp_rectangle and self.temp_rect_timer and
                    time.time() - self.temp_rect_timer <= self.temp_rect_duration):
                cv2.rectangle(display_frame,
                              self.temp_rectangle[0],
                              self.temp_rectangle[1],
                              (255, 255, 255), 1)
            else:
                self.temp_rectangle = None
                self.temp_rect_timer = None

            combined_frame = cv2.addWeighted(display_frame, 1, edge_mask, 0.5, 0)
            cv2.imshow("HighLightAnythingEngine", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = HighlightTracker()
    tracker.run()