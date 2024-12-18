from baseline import BaselineJPEGDecoder
import verify
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def decode_image(filename: str):
    input_path = os.path.join('inputs', filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nProcessing: {input_path}")

    name, _ = os.path.splitext(filename)
    output_path = os.path.join('outputs', f'decoded_{name}.raw')

    decoder = BaselineJPEGDecoder(input_path)

    try:
        decoded_image = decoder.decode()
        plt.imshow(decoded_image)
        plt.show()
        save_image(decoded_image, output_path)
        print(f"Successfully decoded to: {output_path}")
        # verification
        verify.verify_decoder(input_path, output_path)
    except Exception as e:
        print(f"Error during decoding: {str(e)}")


def save_image(image: np.ndarray, output_path: str):
    image.tofile(output_path)


def main():
    parser = argparse.ArgumentParser(description='JPEG Decoder')
    parser.add_argument('--image', type=str, required=True, help='Name of input JPEG image')
    args = parser.parse_args()

    try:
        decode_image(args.image)
    except Exception as e:
        print(f"Error processing {args.image}: {str(e)}")

if __name__ == "__main__":
    main()
