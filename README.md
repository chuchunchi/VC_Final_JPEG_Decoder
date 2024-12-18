# VC_Final_JPEG_Decoder

## Overview

This project implements a baseline JPEG decoder in Python. It reads .jpeg images, decodes them, and outputs raw image data. The results are validated using the Pillow library to compare accuracy.

## Steps to Run the Project

1. Install Requirements

    Before running the program, install the necessary Python libraries:

    ```
    pip install -r requirements.txt
    ```

2. Input an Image

    Place your `.jpeg` or `.jpg` files in the `inputs` folder.

3. Run the Program

    Execute the program by providing an input image file:

    ```
    python3 main.py --image fish.jpeg
    ```

4. Outputs

    After running the program, you will receive:

    - A pop-up window displaying the decoded image.
    - A `.raw` file saved in the `outputs` folder.
    - Comparison Metrics between the decoded image and the Pillow reference image.


## File Structure

```
VC_FINAL_JPEG_DECODER/
│
├── inputs/
│   ├── car.jpeg           # Sample input image 1
│   └── fish.jpeg          # Sample input image 2
│
├── outputs/
│   ├── decoded_car.raw    # Output: Decoded raw image for car.jpeg
│   └── decoded_fish.raw   # Output: Decoded raw image for fish.jpeg
│
├── baseline.py            # Baseline JPEG decoder class implementation
├── codec.py               # Zigzag pattern generator for decoding
├── main.py                # Main program handling arguments and invoking the decoder
├── verify.py              # Verifies decoded image accuracy using Pillow and outputs metrics
│
├── requirements.txt       # List of required Python libraries
```

### File Descriptions
- baseline.py: Contains the implementation of the baseline JPEG decoder class.
- codec.py: Generates the zigzag pattern used in the decoding process.
- main.py: The entry point of the program. Handles arguments and calls the JPEG decoder.
- verify.py: Verifies the decoded image by comparing it to the reference image decoded using Pillow. Outputs comparison metrics such as error rates.


### Results


| Image | fish.jpeg | car.jpeg |
| -------- | -------- | -------- |
| MAE      | 55.858566 | 77.129160 | 
| MSE        | 0.402709  | 2.388816 | 
| RMSE       | 0.634594  | 1.545580 |
| PSNR       | 52.080890 | 44.284812 | 
| SSIM       | 0.996752  | 0.995778 |
| Red Channel MAE    | 59.840596  | 81.042063 | 
| Green Channel MAE  | 39.474734  | 47.361250 | 
| Blue Channel MAE   | 68.260368  | 102.984167 |
