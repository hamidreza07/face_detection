import glob
import os
import cv2
import time
import face_detection


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":
    for item in ['brigh_cont','brigh_cont - rotate','brigh_gray_rotate','gray_scale','gray_scale - rotation','main_rotation']:
        main = item
        impaths = os.listdir(main)
        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold = 0.5,
            nms_iou_threshold =0.005 ,
            max_resolution=1500            
                )
        for impath in impaths:
            if not impath.endswith(".png"): continue
            im = cv2.imread(f"{main}/{impath}")
            print("Processing:", impath)
            t = time.time()
            dets = detector.detect(
                im[:, :, ::-1]
            )[:, :4]
            print(f"Detection time: {time.time()- t:.3f}")
            draw_faces(im, dets)
            imname = os.path.basename(impath).split(".")[0]
            os.makedirs(f'res_{main}',exist_ok=True)
            cv2.imwrite(f"res_{main}/{impath}", im)
            