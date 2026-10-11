import pyarrow.parquet as pq
import pandas as pd

PATH = r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted\Bangladesh\Bangladesh_part_1.gzip.parquet"

pf = pq.ParquetFile(PATH)
print(pf.metadata.num_rows, "rows,", pf.metadata.num_columns, "columns")
print(pf.schema_arrow)

df = next(pf.iter_batches(batch_size=1000)).to_pandas()
print(df.head())

