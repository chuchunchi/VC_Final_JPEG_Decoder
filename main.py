from baseline import BaselineJPEGDecoder
import verify
from progressive import ProgressiveJPEGDecoder
import numpy as np
import os
import argparse

def decode_image(filename: str):

    input_path = os.path.join('inputs', filename)
    # Check if file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    """Decode JPEG image and save the result"""
    # Check the type of JPEG
    result = check_jpeg_type(input_path)
    print(f"\nProcessing: {filename}")
    print(f"Encoding Type: {result['encoding_type']}")
    print(f"Specific Marker: {result['specific_marker']}")

    # Choose appropriate decoder
    if result['encoding_type'] == "Progressive":
        decoder = ProgressiveJPEGDecoder(input_path)
    else:
        decoder = BaselineJPEGDecoder(input_path)

    name, _ = os.path.splitext(filename)
    output_path = os.path.join('outputs', f'decoded_{name}.raw')

    # Decode and save
    try:
        decoded_image = decoder.decode()
        save_image(decoded_image, output_path)
        print(f"Successfully decoded to: {output_path}")
        # verification
        verify.verify_decoder(input_path, output_path)
        return True
    except Exception as e:
        print(f"Error during decoding: {str(e)}")
        return False


def check_jpeg_type(file_path):
    """
    Check if a JPEG file is baseline or progressive by examining its SOF marker.
    Returns the encoding type and marker found.
    """
    # JPEG markers
    SOF_MARKERS = {
        b'\xFF\xC0': "Baseline DCT (SOF0)",
        b'\xFF\xC1': "Extended Sequential DCT (SOF1)",
        b'\xFF\xC2': "Progressive DCT (SOF2)",
        b'\xFF\xC3': "Lossless Sequential (SOF3)",
        b'\xFF\xC5': "Differential Sequential DCT (SOF5)",
        b'\xFF\xC6': "Differential Progressive DCT (SOF6)",
        b'\xFF\xC7': "Differential Lossless (SOF7)",
        b'\xFF\xC9': "Extended Sequential DCT, Arithmetic coding (SOF9)",
        b'\xFF\xCA': "Progressive DCT, Arithmetic coding (SOF10)",
        b'\xFF\xCB': "Lossless, Arithmetic coding (SOF11)",
        b'\xFF\xCD': "Differential Sequential DCT, Arithmetic coding (SOF13)",
        b'\xFF\xCE': "Differential Progressive DCT, Arithmetic coding (SOF14)",
        b'\xFF\xCF': "Differential Lossless, Arithmetic coding (SOF15)",
    }

    with open(file_path, 'rb') as f:
        # Check for JPEG signature
        if f.read(2) != b'\xFF\xD8':
            raise ValueError("Not a valid JPEG file")

        while True:
            marker = f.read(2)
            if marker == b'':  # EOF
                raise ValueError("No SOF marker found")

            # Check if it's an SOF marker
            if marker in SOF_MARKERS:
                return {
                    "encoding_type": "Progressive" if marker in [b'\xFF\xC2', b'\xFF\xCA'] else "Baseline/Sequential",
                    "specific_marker": SOF_MARKERS[marker],
                    "marker_hex": marker.hex().upper()
                }

            # If not SOF, skip this segment
            if marker[0] == 0xFF and marker[1] >= 0xC0:
                length = int.from_bytes(f.read(2), 'big')
                f.seek(length-2, 1)  # Skip the segment data

def save_image(image: np.ndarray, output_path: str):
    """Save decoded image as raw pixels"""
    image.tofile(output_path)

def main():
    parser = argparse.ArgumentParser(description='JPEG Decoder')
    parser.add_argument('--image', type=str, required=True, help='Path to input JPEG image')
    args = parser.parse_args()

    try:
        decode_image(args.image)
    except Exception as e:
        print(f"Error processing {args.image}: {str(e)}")

if __name__ == "__main__":
    main()
