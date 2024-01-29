from argparse import ArgumentParser
import os

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img_dataset', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('out_dataset', help='Image file')
    
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    path = args.img_dataset
    output_path = args.out_dataset
    for item in os.listdir(path):
        img_path = f'{path}/{item}'
        # test a single image
        result = inference_detector(model, img_path)
        # show the results
        os.makedirs(output_path,exist_ok=True)
        show_result_pyplot(model, img_path, result, score_thr=args.score_thr,img_result=f"{output_path}/{item}")


if __name__ == '__main__':
    main()
