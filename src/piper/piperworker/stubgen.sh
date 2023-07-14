stubgen -m cv2 -o $(python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')
stubgen -m websockets -o $(python -c 'import websockets, os; print(os.path.dirname(websockets.__file__))')
