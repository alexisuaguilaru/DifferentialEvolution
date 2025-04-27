import os
import zipfile
from pathlib import Path

def CompressSummaryResults(Dimension:int,PopulationSize:int,VariantNames:list[str]) -> None:
    with zipfile.ZipFile('./Experiments/'+f'{Dimension}_{PopulationSize}_Results.zip','w',zipfile.ZIP_DEFLATED) as file_zip:
        for variant in VariantNames:
            folder_variant = Path('./Experiments/'+variant).resolve()

            for root , _ , files in os.walk(folder_variant):
                for file in files:
                    file_path = Path(root) / file

                    relative_file_path = file_path.relative_to(folder_variant.parent)
                    file_zip.write(file_path,relative_file_path)

                    file_path.unlink()
            
            folder_variant.rmdir()