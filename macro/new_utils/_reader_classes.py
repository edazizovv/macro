#


#
import pandas

#


#
class CSVReader:
    def __init__(self):
        self.extension = ".csv"
        self.kwargs = {
            "sep": ","
        }
    def read(self, source_formatter, name):
        d = f"{source_formatter}{name}{self.extension}"
        frame = pandas.read_csv(d, **self.kwargs)
        return frame
