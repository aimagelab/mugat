import pandas as pd
from urllib.request import urlopen
from shutil import copyfileobj
from sys import argv

df = pd.read_csv(argv[1])

for _i, el in enumerate(df["id"]):
    print(f"downloading {_i}-th element: {el}", flush=True)

    try:
        with urlopen(el.replace('abs', 'src')) as response, open("tex/%d.tar.gz" % _i, 'wb') as out_file:
            copyfileobj(response, out_file)

        with urlopen(el.replace('abs', 'pdf')) as response, open("pdf/%d.pdf" % _i, 'wb') as out_file:
            copyfileobj(response, out_file)
    except:
        print("skipped")

    

