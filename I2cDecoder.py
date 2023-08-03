#! /bin/python

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class I2cDecoder:
    def __init__(self, hmin=1.0, hmax=2.0, use_ack=True) -> None:
        self.hmin = hmin
        self.hmax = hmax
        self.sampled_df = None
        self.decoded = None
        self.use_ack = use_ack

        # decoding states
        self.symbols = []
        self.timestamp_symbols = []
        self.last_sda = 1
        self.last_scl = 1
        self.started = False
        self.sda_sampling_buffer = [] # holds the SDA samples when SCL is idle
        self.bits_buffer = [] # holds the bits up to 8
        self.wait_for_ack = False

        pass

    def _sample_digital_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        sda = 1
        scl = 1

        df_copy = df.copy()
        
        for irow, row in df_copy.iterrows():
            if row[df_copy.columns[1]] < self.hmin:
                sda = 0
            elif row[df_copy.columns[1]] > self.hmax:
                sda = 1
            
            if row[df_copy.columns[2]] < self.hmin:
                scl = 0
            elif row[df_copy.columns[2]] > self.hmax:
                scl = 1

            row[df_copy.columns[1]] = sda
            row[df_copy.columns[2]] = scl

        return df_copy

    def _run_events_decoder(self, sda: int, scl: int, time: float) -> None:
        do_return = False
        
        if self.last_scl == 1 and scl == 1 and self.last_sda == 1 and sda == 0:

            self.bits_buffer.clear()
            self.sda_sampling_buffer.clear()
            if self.started:
                # repeated start condition
                self.symbols.append("Sr")
                self.timestamp_symbols.append(time)

                do_return = True
            else:
                # start condition
                self.symbols.append("S")
                self.timestamp_symbols.append(time)
                self.started = True
                do_return = True

        elif self.last_scl == 1 and scl == 1 and self.last_sda == 0 and sda == 1:
            # stop condition
            self.symbols.append("P")
            self.timestamp_symbols.append(time)
            self.started = False
            do_return = True

        if not self.started or do_return:
            self.last_sda = sda
            self.last_scl = scl
            return


        if scl == 1:
            # SCL is sampling the SDA
            self.sda_sampling_buffer.append(sda)

        elif len(self.sda_sampling_buffer) == 0:
            # CLK is still low after the bit was decoded
            self.last_sda = sda
            self.last_scl = scl
            return

        else:
            # sample right in the middle of the clock pulse.
            # maybe averaging is also a valid strategy (although more computation intensive)
            middle_index = int(len(self.sda_sampling_buffer)/2)

            sda_sample: int = self.sda_sampling_buffer[middle_index]
            self.sda_sampling_buffer.clear()

            # if previous clock pulse read all 8 bits, it should wait for the ACK bit
            if self.wait_for_ack and self.use_ack:
                self.symbols.append("ACK" if 0 == sda_sample else "NACK")
                self.timestamp_symbols.append(time)
                self.wait_for_ack = False
                self.bits_buffer.clear()

            self.bits_buffer.append(sda_sample)


            byte = 0
            if len(self.bits_buffer) > 8:
                # since I²C transmits MSB first, invert the buffer order to compose the byte by shifting the bits
                inverted_byte_buffer = self.bits_buffer[::-1]
                for bit_index, bit in enumerate(inverted_byte_buffer):
                    byte |= int(bit) << int(bit_index)

                self.bits_buffer.clear()

                # when the byte is received, wait for ACK on the next clock pulse
                self.wait_for_ack = True
            
                self.symbols.append(byte)
                self.timestamp_symbols.append(time)

        self.last_sda = sda
        self.last_scl = scl
        return

    def decode(self, df: pd.DataFrame) -> list:
        """Decodes a Pandas DataFrame formatted as:
        time | SDA  | SCL
        -----|------|------
        0    | 3.22 | 3.21 
        1e-6 | 3.22 | 3.21 
        ...

        After decoding, a call to plot_decoded is suggested

        Args:
            df (pd.DataFrame): DataFrame with time, ch1 and ch2 columns

        Returns:
            list: list of the decoded symbols
        """
        self.sampled_df = self._sample_digital_signal(df)

        time = np.array(self.sampled_df[self.sampled_df.columns[0]])
        sda_stream = np.array(self.sampled_df[self.sampled_df.columns[1]])
        scl_stream = np.array(self.sampled_df[self.sampled_df.columns[2]])

        assert len(sda_stream) == len(scl_stream), "SDA and SCL streams must have the same size" 
        for i, _ in enumerate(scl_stream):
            self._run_events_decoder(sda_stream[i], scl_stream[i], time[i])
        
        return self.symbols
        
        

    def plot_decoded(self, show=False) -> plt.Figure:
        """Plots the decoded Pandas DataFrame

        Args:
            show (bool, optional): If True, shows the plot. Defaults to False.

        Raises:
            SyntaxError: _description_

        Returns:
            plt.Figure: _description_
        """
        if self.sampled_df is None:
            raise SyntaxError("Cannot plot before calling `decode`!")

        fig = plt.figure(dpi=150, figsize=(100,4))

        columns = self.sampled_df.columns[:3]
        time, ch1, ch2 = columns

        plt.plot(self.sampled_df[time], self.sampled_df[ch1], label="SDA");
        plt.plot(self.sampled_df[time], self.sampled_df[ch2], label="SCL");

        plt.legend()
        assert len(self.timestamp_symbols) == len(self.symbols), "Not all symbols have a timestamp!"

        y_offsets = (1.05, 1.10, 1.15)
        which_y = 0
        for i, sym in enumerate(self.symbols):
            y_offset = y_offsets[which_y]
            which_y = (which_y + 1) % len(y_offsets)

            plt.text(self.timestamp_symbols[i], y_offset, self.symbols[i])
            plt.plot([self.timestamp_symbols[i], self.timestamp_symbols[i]], [y_offset, 1], ":k", linewidth=0.5)

        plt.xlim([self.timestamp_symbols[0], 1.01*self.timestamp_symbols[-1]]);
        
        if show:
            plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Pass the CSV file as argument!\n  E.g. `$ ./I2cDecoder.py file.csv`")
        print("\nMake sure the CSV is formatted as:\nTime,SDA,SCL\nt0,x1(t0),x2(t0)\nt1,x1(t1),x2(t1)\n...")
        exit(1)

    csv_file = sys.argv[1]


    df = pd.read_csv(csv_file)
    df = df[df.columns[:3]]

    i2c_decoder = I2cDecoder()

    decoded_symbols = i2c_decoder.decode(df)

    fig = i2c_decoder.plot_decoded(show=True)

