"""ROI selection tool for interactive region selection.

Provides interactive ROI selection using OpenCV mouse callbacks.
Usage:
    selector = ROISelector()
    roi = selector.select_roi(image)
    # roi: [(x1, y1), (x2, y2)]
"""
import logging
import cv2

logger = logging.getLogger(__name__)


class ROISelector:
    """Interactive ROI selector using mouse drag."""
    
    def __init__(self):
        self.roi_box = []
        self.drawing = False
        self._image = None
        self._window_name = "Select ROI"

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_box = [(x, y)]
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self._image.copy()
            cv2.rectangle(img_copy, self.roi_box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(self._window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_box.append((x, y))
            self.drawing = False

    def select_roi(self, image: cv2.typing.MatLike, example_image_path: str | None = None) -> tuple | None:
        """
        Select ROI interactively.
        
        Args:
            image: Base image for selection
            example_image_path: Optional path to example image to display
            
        Returns:
            tuple: [(x1, y1), (x2, y2)] normalized coordinates, or None if cancelled
        """
        self._image = image.copy()
        
        while True:
            # Add instruction text
            cv2.putText(
                self._image,
                "Drag to select area | ENTER:confirm | ESC:retry",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Show windows
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, 1280, 720)
            cv2.setMouseCallback(self._window_name, self.mouse_callback, self._image)
            cv2.imshow(self._window_name, self._image)

            # Show example if provided
            if example_image_path and cv2.imdecode is not None:
                example_img = cv2.imread(example_image_path)
                if example_img is not None:
                    cv2.imshow("example", example_img)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == 13 and len(self.roi_box) == 2:  # Enter
                # Normalize coordinates
                x1 = min(self.roi_box[0][0], self.roi_box[1][0])
                y1 = min(self.roi_box[0][1], self.roi_box[1][1])
                x2 = max(self.roi_box[0][0], self.roi_box[1][0])
                y2 = max(self.roi_box[0][1], self.roi_box[1][1])
                logger.info(f"Selected ROI: {[(x1, y1), (x2, y2)]}")
                return [(x1, y1), (x2, y2)]
            elif key == 27:  # ESC
                self.roi_box = []
                continue
            else:
                self.roi_box = []
                continue
