stubgen -m cv2 -o $(python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')
