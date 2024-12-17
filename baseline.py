import numpy as np
from typing import List, Tuple, Dict, BinaryIO
from collections import defaultdict
import math

# Zigzag Order for traversing 8x8 blocks
zigzag = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
          (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
          (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
          (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
          (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
          (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
          (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
          (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

# Class to represent a Huffman Table
class HuffmanTable:
    table_class: int
    dest_id: int
    counts: List[int]
    huff_data: Dict[Tuple[int, int], int]

# Class to represent a Quantization Table
class QuantizationTable:
    precision: int
    dest_id: int
    table: List[List[int]]

# Class to represent a Frame Component
class FrameComponent:
    identifier: int
    sampling_factor: int
    h_sampling_factor: int
    v_sampling_factor: int
    quant_table_dest: int

# Class to represent the Start of Frame (SOF) segment
class StartOfFrame:
    precision: int
    num_lines: int
    samples_per_line: int
    components: List[FrameComponent]

# Class to represent a Scan Component
class ScanComponent:
    selector: int
    dc_table: int
    ac_table: int

# Class to represent the Start of Scan (SOS) segment
class StartOfScan:
    components: List[ScanComponent]
    spectral_selection_range: Tuple[int, int]
    successive_approximation: int

# Main class for decoding baseline JPEG images
class BaselineJPEGDecoder:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.huffman_tables = {}
        self.quantization_tables = {}
        self.sos = None
        self.sof = None
        self.idct_matrix = self._build_idct_matrix()
        self.scan_data = None

    def decode(self) -> np.ndarray:
        """
        Main decoding process for baseline JPEG.

        This method reads the JPEG file, processes its segments, and decodes the image data into an RGB format.

        Returns:
            np.ndarray: The decoded image in RGB format.

        Raises:
            ValueError: If an invalid JPEG segment is encountered.

        The decoding process involves:
        - Reading the JPEG file and identifying segments.
        - Parsing headers for Start of Frame (SOF), Start of Scan (SOS), Define Quantization Table (DQT), and Define Huffman Table (DHT).
        - Handling the scan data and performing Huffman decoding.
        - Performing Inverse Discrete Cosine Transform (IDCT) on the decoded data.
        - Converting YCbCr color space to RGB.
        """
        with open(self.filepath, 'rb') as f:
            block_id_bytes = f.read(2)

            while block_id_bytes:
                block_id = int.from_bytes(block_id_bytes, byteorder="big")

                if block_id < 0xFFC0:  # JPEG Segment Error
                    raise ValueError(f"Invalid JPEG file - unexpected block ID: {block_id:04X}")

                if block_id == 0xFFDA:  # Start of Scan (SOS)
                    print("Start of Scan")
                    self.sos = self._parse_scan_header(f)
                    assert self.sof is not None

                    max_h_sampling_factor = 0
                    max_v_sampling_factor = 0
                    for component in self.sof.components:
                        max_h_sampling_factor = max(max_h_sampling_factor, component.h_sampling_factor)
                        max_v_sampling_factor = max(max_v_sampling_factor, component.v_sampling_factor)

                    mcu_size_x = 8 * max_h_sampling_factor
                    mcu_size_y = 8 * max_v_sampling_factor

                    num_mcu_x = math.ceil(self.sof.samples_per_line / mcu_size_x)
                    num_mcu_y = math.ceil(self.sof.num_lines / mcu_size_y)

                    start_pos = f.tell()
                    f.seek(0, 2)
                    end_pos = f.tell()

                    f.seek(start_pos)
                    self.scan_data = bytearray(f.read(end_pos - start_pos))

                    marker_pos = None
                    marker_pos_diff = 0

                    for i in range(len(self.scan_data) - 2, 0, -1):
                        marker_code = self.scan_data[i:i + 2]

                        if marker_code == b'\xFF\x00':
                            self.scan_data.pop(i + 1)
                            marker_pos_diff += 1

                        elif marker_code > b'\xFF\x00':
                            if b'\xFF\xD0' <= marker_code <= b'\xFF\xD7':
                                self.scan_data.pop(i)
                                marker_pos_diff += 1
                            else:
                                marker_pos = i
                                marker_pos_diff = 0

                    assert marker_pos is not None

                    self.scan_data = self.scan_data[:marker_pos - marker_pos_diff]
                    f.seek(start_pos + marker_pos)

                    # Initialize RGB image
                    image_rgb = [[(0, 0, 0) for _ in range(self.sof.samples_per_line)] for _ in range(self.sof.num_lines)]

                    curr_bit = 0

                    predictions = [0 for _ in range(len(self.sos.components))]

                    for mcu_row in range(num_mcu_y):
                        for mcu_col in range(num_mcu_x):
                            mcu_arr = []

                            for component_idx, component in enumerate(self.sos.components):
                                frame_component = None
                                for c in self.sof.components:
                                    if c.identifier == component.selector:
                                        frame_component = c
                                        break
                                assert frame_component is not None

                                quant_table = self.quantization_tables[frame_component.quant_table_dest]
                                dc_huff_table = self.huffman_tables[component.dc_table, 0]
                                ac_huff_table = self.huffman_tables[component.ac_table, 1]


                                mcu = [[0 for _ in range(8 * frame_component.h_sampling_factor)] for _ in range(8 * frame_component.v_sampling_factor)]

                                for data_unit_row in range(frame_component.v_sampling_factor):
                                    for data_unit_col in range(frame_component.h_sampling_factor):

                                        dc_code, length = self._get_next_huffman_value(self.scan_data, curr_bit, dc_huff_table)
                                        curr_bit += length

                                        additional_bits = self._bits_from_bytearray(self.scan_data, curr_bit, dc_code, "big")
                                        curr_bit += dc_code

                                        diff = self._get_signed_value(additional_bits, dc_code)
                                        abs_dc_value = predictions[component_idx] + diff

                                        predictions[component_idx] = abs_dc_value

                                        dct_coeffs = [0 for _ in range(64)]
                                        dct_coeffs[0] = abs_dc_value

                                        k = 0
                                        while k != 63:
                                            k += 1

                                            rs, length = self._get_next_huffman_value(self.scan_data, curr_bit, ac_huff_table)
                                            curr_bit += length

                                            rrrr = rs >> 4
                                            ssss = rs & 0b1111

                                            if ssss == 0:
                                                if rrrr == 15:
                                                    k += 15
                                                    continue
                                                else:
                                                    break

                                            k += rrrr

                                            additional_bits = self._bits_from_bytearray(self.scan_data, curr_bit, ssss, "big")
                                            curr_bit += ssss

                                            v = self._get_signed_value(additional_bits, ssss)

                                            dct_coeffs[k] = v

                                        dct_matrix = [[0 for _ in range(8)] for _ in range(8)]
                                        for i, coeff in enumerate(dct_coeffs):
                                            row, col = zigzag[i]
                                            dct_matrix[row][col] = coeff * quant_table[row][col]

                                        for y in range(8):
                                            for x in range(8):
                                                val = 0
                                                for u in range(8):
                                                    for v in range(8):
                                                        val += self.idct_matrix[y][x][u][v] * dct_matrix[v][u]
                                                val /= 4

                                                mcu[(data_unit_row * 8) + y][(data_unit_col * 8) + x] = val

                                horiz_multiplier = max_h_sampling_factor // frame_component.h_sampling_factor
                                vert_multiplier = max_v_sampling_factor // frame_component.v_sampling_factor

                                if vert_multiplier > 1 or horiz_multiplier > 1:
                                    mcu = [[val for val in row for _ in range(horiz_multiplier)] for row in mcu for _ in range(vert_multiplier)]
                                mcu_arr.append(mcu)

                            for i in range(mcu_size_y):
                                if (mcu_row * mcu_size_y) + i >= self.sof.num_lines:
                                    break

                                for j in range(mcu_size_x):
                                    if (mcu_col * mcu_size_x) + j >= self.sof.samples_per_line:
                                        break

                                    image_rgb[(mcu_row * mcu_size_y) + i][(mcu_col * mcu_size_x) + j] = self._ycbcr_to_rgb(mcu_arr[0][i][j], mcu_arr[1][i][j], mcu_arr[2][i][j])

                elif block_id == 0xFFD8:  # Start of Image (SOI)
                    print("Start of Image")
                    pass
                elif block_id == 0xFFC0 or block_id == 0xFFC1:  # Start of Frame (SOF)
                    print("Start of Frame")
                    self.sof = self._parse_frame_header(f)
                elif block_id == 0xFFDB:  # Define Quantization Table (DQT)
                    print("Define Quantization Table")
                    for table in self._parse_quantization_tables(f):
                        self.quantization_tables[table.dest_id] = table.table
                elif block_id == 0xFFC4:  # Define Huffman Table (DHT)
                    print("Define Huffman Table")
                    for table in self._parse_huffman_tables(f):
                        self.huffman_tables[table.dest_id, table.table_class] = table.huff_data
                elif block_id == 0xFFD9:  # End of Image (EOI)
                    print("End of Image")
                    return np.array(image_rgb, dtype=np.uint8)
                else:  # Any other segment
                    size = int.from_bytes(f.read(2), byteorder="big")
                    f.seek(size - 2, 1)

                block_id_bytes = f.read(2)

    def _parse_scan_header(self, file: BinaryIO) -> StartOfScan:
        """
        Parses the Start of Scan (SOS) header from a JPEG file.

        Args:
            file (BinaryIO): The binary file object to read the SOS header from.

        Returns:
            StartOfScan: An object containing the parsed SOS header information, including:
            - num_scan_components (int): Number of components in the scan.
            - components (List[ScanComponent]): List of scan components, each with:
                - selector (int): Component selector.
                - dc_table (int): DC entropy coding table selector.
                - ac_table (int): AC entropy coding table selector.
            - spectral_selection_range (Tuple[int, int]): Start and end of spectral selection.
            - successive_approximation (int): Successive approximation bit position.
        """
        sos = StartOfScan()
        sos.components = []
        size = int.from_bytes(file.read(2), byteorder="big")
        sos.num_scan_components = int.from_bytes(file.read(1), byteorder="big")

        for i in range(sos.num_scan_components):
            component = ScanComponent()
            component.selector = int.from_bytes(file.read(1), byteorder="big")
            temp = int.from_bytes(file.read(1), byteorder="big")

            component.dc_table = temp >> 4
            component.ac_table = temp & 0b1111
            sos.components.append(component)

        spectral_selection_start = int.from_bytes(file.read(1), byteorder="big")
        spectral_selection_end = int.from_bytes(file.read(1), byteorder="big")
        sos.spectral_selection_range = (spectral_selection_start, spectral_selection_end)
        sos.successive_approximation = int.from_bytes(file.read(1), byteorder="big")

        return sos

    def _parse_frame_header(self, file: BinaryIO) -> StartOfFrame:
        """
        Parses the frame header from a binary file stream.

        Args:
            file (BinaryIO): The binary file stream to read the frame header from.

        Returns:
            StartOfFrame: An object containing the parsed frame header information, including:
            - precision: The sample precision.
            - num_lines: The number of lines in the frame.
            - samples_per_line: The number of samples per line.
            - num_frame_components: The number of components in the frame.
            - components: A list of FrameComponent objects, each containing:
                - identifier: The component identifier.
                - h_sampling_factor: The horizontal sampling factor.
                - v_sampling_factor: The vertical sampling factor.
                - quant_table_dest: The destination of the quantization table.
        """
        sof = StartOfFrame()
        sof.components = []
        size = int.from_bytes(file.read(2), byteorder="big")

        sof.precision = int.from_bytes(file.read(1), byteorder="big")
        sof.num_lines = int.from_bytes(file.read(2), byteorder="big")
        sof.samples_per_line = int.from_bytes(file.read(2), byteorder="big")
        sof.num_frame_components = int.from_bytes(file.read(1), byteorder="big")

        for i in range(sof.num_frame_components):
            component = FrameComponent()
            component.identifier = int.from_bytes(file.read(1), byteorder="big")
            component.sampling_factor = int.from_bytes(file.read(1), byteorder="big")
            component.h_sampling_factor = component.sampling_factor >> 4
            component.v_sampling_factor = component.sampling_factor & 0b1111
            component.quant_table_dest = int.from_bytes(file.read(1), byteorder="big")
            sof.components.append(component)

        return sof

    def _parse_quantization_tables(self, file: BinaryIO) -> List[QuantizationTable]:
        """
        Parses the quantization tables from the given binary file.

        Args:
            file (BinaryIO): The binary file to read the quantization tables from.

        Returns:
            List[QuantizationTable]: A list of parsed quantization tables.

        The function reads the size of the quantization tables, then iterates through the file to read each table.
        Each table is read based on its precision and destination ID, and the elements are stored in a zigzag order.
        """
        size = int.from_bytes(file.read(2), byteorder="big")
        quant_tables = []
        bytes_left = size - 2

        while bytes_left > 0:
            quant_table = QuantizationTable()
            quant_table.table = {}
            quant_table.table = [[0 for _ in range(8)] for _ in range(8)]

            temp = int.from_bytes(file.read(1), byteorder="big")
            quant_table.precision = temp >> 4
            element_bytes = 1 if quant_table.precision == 0 else 2
            quant_table.dest_id = temp & 0b1111
            bytes_left -= 65 + (64 * quant_table.precision)

            for i in range(64):
                element = int.from_bytes(file.read(element_bytes), byteorder="big")
                row, col = zigzag[i]
                quant_table.table[row][col] = element

            quant_tables.append(quant_table)

        return quant_tables

    def _parse_huffman_tables(self, file: BinaryIO) -> List[HuffmanTable]:
        """
        Parses Huffman tables from a binary file stream.

        Args:
            file (BinaryIO): A binary file stream to read the Huffman tables from.

        Returns:
            List[HuffmanTable]: A list of HuffmanTable objects parsed from the file.

        The function reads the size of the Huffman tables, then iterates through the
        file to read each Huffman table. Each table's class, destination ID, and counts
        for each code length are extracted. The Huffman data is then populated with
        the corresponding codes and their lengths. The function continues until all
        bytes specified by the size are read.
        """
        size = int.from_bytes(file.read(2), byteorder="big")
        huff_tables = []
        bytes_left = size - 2

        while bytes_left > 0:
            huff_table = HuffmanTable()
            huff_table.huff_data = {}

            table = int.from_bytes(file.read(1), byteorder="big")
            huff_table.table_class = table >> 4
            huff_table.dest_id = table & 0b1111

            huff_table.counts = [int.from_bytes(file.read(1), byteorder="big") for _ in range(16)]

            length_codes_map = defaultdict(list)
            code = 0
            for i in range(16):
                for j in range(huff_table.counts[i]):
                    huff_byte = int.from_bytes(file.read(1), byteorder="big")
                    huff_table.huff_data[(code, i+1)] = huff_byte
                    length_codes_map[i+1].append(huff_byte)
                    code += 1
                code <<= 1

            bytes_left -= 17 + sum(huff_table.counts)
            huff_tables.append(huff_table)

        return huff_tables

    def _bit_from_bytearray(self, arr: bytearray, bit_idx: int, order: str) -> int:
        """
        Extracts a specific bit from a bytearray.

        Args:
            arr (bytearray): The bytearray from which to extract the bit.
            bit_idx (int): The index of the bit to extract.
            order (str): The bit order, either 'little' for little-endian or 'big' for big-endian.

        Returns:
            int: The extracted bit (0 or 1).
        """
        if order == "little":
            return (arr[bit_idx // 8] & (0b1 << (bit_idx % 8))) >> (bit_idx % 8)
        else:
            return (arr[bit_idx // 8] & (0b1 << (7 - (bit_idx % 8)))) >> (7 - (bit_idx % 8))

    def _bits_from_bytearray(self, arr: bytearray, start_idx: int, num_bits: int, order: str) -> int:
        """
        Extracts a sequence of bits from a bytearray and returns it as an integer.

        Args:
            arr (bytearray): The bytearray to extract bits from.
            start_idx (int): The starting index (in bits) from which to begin extraction.
            num_bits (int): The number of bits to extract.
            order (str): The bit order, either 'big' or 'little'.

        Returns:
            int: The extracted bits as an integer.
        """
        out = 0
        for bit_idx in range(start_idx, start_idx + num_bits):
            out = (out << 1) | self._bit_from_bytearray(arr, bit_idx, order)
        return out

    def _get_next_huffman_value(self, data: bytearray, data_pos: int, huff_table: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
        """
        Retrieve the next Huffman value from the given data using the provided Huffman table.

        Args:
            data (bytearray): The bytearray containing the encoded data.
            data_pos (int): The current position in the data bytearray.
            huff_table (Dict[Tuple[int, int], int]): The Huffman table mapping encoded bit sequences to their corresponding values.

        Returns:
            Tuple[int, int]: A tuple containing the decoded Huffman value and the number of bits used to decode it.
        """
        encoded_bits = self._bit_from_bytearray(data, data_pos, "big")
        start_bit = data_pos
        curr_pos = data_pos + 1

        while (encoded_bits, curr_pos - start_bit) not in huff_table:
            encoded_bits = (encoded_bits << 1) | self._bit_from_bytearray(self.scan_data, curr_pos, "big")
            curr_pos += 1

        num_bits = curr_pos - start_bit

        return huff_table[(encoded_bits, num_bits)], num_bits

    def _get_signed_value(self, bits: int, num_bits: int) -> int:
        """
        Converts an unsigned integer to a signed integer using the specified number of bits.

        Args:
            bits (int): The unsigned integer value to be converted.
            num_bits (int): The number of bits representing the integer.

        Returns:
            int: The signed integer value.
        """
        if bits < 2 ** (num_bits - 1):
            min_val = (-1 << num_bits) + 1
            return min_val + bits
        return bits

    def _build_idct_matrix(self) -> List:
        """
        Builds the Inverse Discrete Cosine Transform (IDCT) matrix.

        This method generates an 8x8 matrix used for the IDCT process in JPEG decoding.
        The matrix is precomputed to optimize the IDCT calculations for each pixel block.

        Returns:
            List: A 3-dimensional list representing the IDCT coefficients for an 8x8 block.
        """
        idct_lookup = []
        for y in range(8):
            idct_row = []
            for x in range(8):
                uv_matrix = []
                for u in range(8):
                    uv_row = []
                    for v in range(8):
                        cu = (1 / math.sqrt(2)) if u == 0 else 1
                        cv = (1 / math.sqrt(2)) if v == 0 else 1
                        uv_row.append(cu * cv *
                                    math.cos(((2 * x + 1) * u * math.pi) / 16) * math.cos(((2 * y + 1) * v * math.pi) / 16))
                    uv_matrix.append(uv_row)
                idct_row.append(uv_matrix)
            idct_lookup.append(idct_row)
        return idct_lookup

    def _ycbcr_to_rgb(self, lum, chrom_blue, chrom_red) -> Tuple[int, int, int]:
        """
        Converts YCbCr color space values to RGB color space values.

        Parameters:
        lum (float): Luminance component (Y).
        chrom_blue (float): Chrominance blue component (Cb).
        chrom_red (float): Chrominance red component (Cr).

        Returns:
        Tuple[int, int, int]: A tuple containing the RGB values (red, green, blue) each in the range [0, 255].
        """
        red = chrom_red * (2 - 2 * 0.299) + lum
        blue = chrom_blue * (2 - 2 * 0.114) + lum
        green = (lum - 0.114 * blue - 0.299 * red) / 0.587
        return min(255, max(0, round(red + 128))), min(255, max(0, round(green + 128))), min(255, max(0, round(blue + 128)))
