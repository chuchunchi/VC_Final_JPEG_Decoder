import numpy as np
from typing import List, Tuple, Dict
import struct

class BaselineJPEGDecoder:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.quantization_tables = {}
        self.huffman_tables = {'dc': {}, 'ac': {}}
        self.components = {}
        self.image_height = 0
        self.image_width = 0
        self.mcu_size = (8, 8)
        self.idct_matrix = self._build_idct_matrix()

    def decode(self) -> np.ndarray:
        """Main decoding process for baseline JPEG"""
        with open(self.filepath, 'rb') as f:
            if f.read(2) != b'\xFF\xD8':
                raise ValueError("Invalid JPEG file - missing SOI marker")

            while True:
                marker = f.read(2)
                if marker == b'':
                    break

                if marker[0] != 0xFF:
                    raise ValueError("Invalid marker format")

                if marker == b'\xFF\xD9':  # EOI marker
                    break

                if marker == b'\xFF\xDB':  # DQT marker
                    self._parse_quantization_tables(f)
                elif marker == b'\xFF\xC0':  # SOF0 marker
                    self._parse_frame_header(f)
                elif marker == b'\xFF\xC4':  # DHT marker
                    self._parse_huffman_tables(f)
                elif marker == b'\xFF\xDA':  # SOS marker
                    self._parse_scan_header(f)
                    image_data = self._decode_scan_data(f)
                    return image_data
                else:
                    # Skip other markers
                    length = struct.unpack('>H', f.read(2))[0]
                    f.seek(length-2, 1)

        raise ValueError("Incomplete JPEG file")

    def _parse_quantization_tables(self, f) -> None:
        length = struct.unpack('>H', f.read(2))[0]
        bytes_read = 2

        while bytes_read < length:
            table_info = f.read(1)[0]
            precision = table_info >> 4
            table_id = table_info & 0x0F

            if precision == 0:
                table = np.frombuffer(f.read(64), dtype=np.uint8).reshape((8, 8))
                bytes_read += 65
            else:
                table = np.frombuffer(f.read(128), dtype=np.uint16).reshape((8, 8))
                bytes_read += 129

            self.quantization_tables[table_id] = table.astype(np.float32)

    def _parse_frame_header(self, f) -> None:
        length = struct.unpack('>H', f.read(2))[0]
        precision = f.read(1)[0]

        if precision != 8:
            raise ValueError("Only 8-bit precision is supported")

        self.image_height, self.image_width = struct.unpack('>HH', f.read(4))
        num_components = f.read(1)[0]

        for _ in range(num_components):
            component_id = f.read(1)[0]
            sampling_factors = f.read(1)[0]
            h_factor = sampling_factors >> 4
            v_factor = sampling_factors & 0x0F
            quant_table_id = f.read(1)[0]

            self.components[component_id] = {
                'h_factor': h_factor,
                'v_factor': v_factor,
                'quant_table_id': quant_table_id,
                'dc_pred': 0  # Initialize DC predictor
            }

    def _parse_huffman_tables(self, f) -> None:
        """Parse Huffman tables from DHT segment"""
        length = struct.unpack('>H', f.read(2))[0]
        bytes_read = 2

        while bytes_read < length:
            table_info = f.read(1)[0]
            table_class = 'dc' if (table_info >> 4) == 0 else 'ac'
            table_id = table_info & 0x0F

            num_codes = np.frombuffer(f.read(16), dtype=np.uint8)
            bytes_read += 17

            huffval = []
            total_codes = sum(num_codes)
            huffval = list(f.read(total_codes))
            bytes_read += total_codes

            self.huffman_tables[table_class][table_id] = self._build_huffman_table(num_codes, huffval)

    def _build_huffman_table(self, num_codes: np.ndarray, huffval: List[int]) -> Dict:
        huffman_table = {}
        code = 0
        pos = 0

        for i in range(len(num_codes)):
            for _ in range(num_codes[i]):
                huffman_table[format(code, f'0{i+1}b')] = huffval[pos]
                pos += 1
                code += 1
            code <<= 1

        return huffman_table

    def _parse_scan_header(self, f) -> None:
        length = struct.unpack('>H', f.read(2))[0]
        num_components = f.read(1)[0]

        for _ in range(num_components):
            component_id = f.read(1)[0]
            tables = f.read(1)[0]
            dc_table_id = tables >> 4
            ac_table_id = tables & 0x0F

            self.components[component_id].update({
                'dc_table_id': dc_table_id,
                'ac_table_id': ac_table_id
            })

        # Skip 3 bytes
        f.seek(3, 1)

    def _decode_scan_data(self, f) -> np.ndarray:
        # Read compressed data
        compressed_data = bytearray()
        while True:
            byte = f.read(1)[0]
            if byte == 0xFF:
                next_byte = f.read(1)[0]
                if next_byte == 0x00:
                    compressed_data.append(0xFF)
                elif next_byte == 0xD9:  # EOI marker
                    break
                else:
                    raise ValueError(f"Unexpected marker in scan data: FF{next_byte:02X}")
            else:
                compressed_data.append(byte)

        # Convert to bit stream
        bits = ''.join(format(b, '08b') for b in compressed_data)
        pos = 0

        # Initialize output image array
        height_padded = ((self.image_height + 7) // 8) * 8
        width_padded = ((self.image_width + 7) // 8) * 8
        image = np.zeros((height_padded, width_padded, 3), dtype=np.float32)

        # Process each MCU
        try:
            for y in range(0, height_padded, 8):
                for x in range(0, width_padded, 8):
                    # Process each component
                    for comp_id, comp in self.components.items():
                        block = np.zeros((8, 8), dtype=np.float32)

                        # Get Huffman and quantization tables
                        dc_table = self.huffman_tables['dc'][comp['dc_table_id']]
                        ac_table = self.huffman_tables['ac'][comp['ac_table_id']]
                        quant_table = self.quantization_tables[comp['quant_table_id']]

                        # Decode DC coefficient
                        code = ''
                        while code not in dc_table and pos < len(bits):
                            code += bits[pos]
                            pos += 1

                        if code in dc_table:
                            dc_size = dc_table[code]
                            if dc_size > 0:
                                dc_bits = bits[pos:pos+dc_size]
                                pos += dc_size
                                if dc_bits[0] == '1':
                                    dc_value = int(dc_bits, 2)
                                else:
                                    dc_value = -(int(''.join('1' if b == '0' else '0' for b in dc_bits), 2) + 1)
                            else:
                                dc_value = 0

                            comp['dc_pred'] += dc_value
                            block[0, 0] = float(comp['dc_pred']) * quant_table[0, 0]

                        # Decode AC coefficients
                        k = 1
                        while k < 64:
                            code = ''
                            while code not in ac_table and pos < len(bits):
                                code += bits[pos]
                                pos += 1

                            if code not in ac_table:
                                break

                            rs = ac_table[code]
                            r = rs >> 4
                            s = rs & 0x0F

                            if rs == 0:  # EOB
                                break
                            elif rs == 0xF0:  # ZRL
                                k += 16
                                continue

                            k += r
                            if k >= 64:
                                break

                            if s > 0:
                                ac_bits = bits[pos:pos+s]
                                pos += s
                                if ac_bits[0] == '1':
                                    ac_value = int(ac_bits, 2)
                                else:
                                    ac_value = -(int(''.join('1' if b == '0' else '0' for b in ac_bits), 2) + 1)

                                zz_pos = self._zigzag_position(k)
                                block[zz_pos] = float(ac_value) * quant_table[zz_pos]

                            k += 1

                        # Apply IDCT
                        block = self._inverse_dct(block)

                        # Store in the appropriate color channel
                        if comp_id == 1:  # Y
                            image[y:y+8, x:x+8, 0] = block
                        elif comp_id == 2:  # Cb
                            image[y:y+8, x:x+8, 1] = block
                        elif comp_id == 3:  # Cr
                            image[y:y+8, x:x+8, 2] = block

        except Exception as e:
            print(f"Error during decoding at position {pos}: {str(e)}")
            raise

        # Convert to RGB and return
        return self._ycbcr_to_rgb(image[:self.image_height, :self.image_width])

    def _zigzag_position(self, k: int) -> Tuple[int, int]:
        zigzag_map = [
            (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
            (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
            (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
            (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
            (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
            (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
            (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
            (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
        ]
        return zigzag_map[k]

    def _build_idct_matrix(self) -> np.ndarray:
        """Build the IDCT transformation matrix"""
        A = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    A[i, j] = 1.0 / np.sqrt(8)
                else:
                    A[i, j] = np.cos((2*j + 1) * i * np.pi / 16) * np.sqrt(2.0/8)
        return A

    def _inverse_dct(self, block: np.ndarray) -> np.ndarray:
        """
        Apply inverse DCT to 8x8 block using matrix multiplication
        This is a more accurate and efficient implementation
        """
        # Ensure block is float32 for numerical accuracy
        block = block.astype(np.float32)

        # Apply IDCT: result = A * block * A^T
        temp = np.matmul(self.idct_matrix.T, block)
        result = np.matmul(temp, self.idct_matrix)

        return result

    def _ycbcr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert YCbCr image to RGB color space"""
        # Ensure we're working with float32
        image = image.astype(np.float32)

        # Extract components
        y = image[:, :, 0]
        cb = image[:, :, 1]
        cr = image[:, :, 2]

        # Initialize output array
        rgb = np.zeros_like(image, dtype=np.float32)

        # Convert to RGB
        rgb[:, :, 0] = y + 1.402 * (cr - 128)  # R
        rgb[:, :, 1] = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)  # G
        rgb[:, :, 2] = y + 1.772 * (cb - 128)  # B

        # Level shift and clamp
        rgb = np.clip(rgb + 128, 0, 255)

        return rgb.astype(np.uint8)
