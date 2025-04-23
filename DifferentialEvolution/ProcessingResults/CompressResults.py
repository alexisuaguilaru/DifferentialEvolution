import os
import zipfile

def CompressSummaryResults(Dimension:int,PopulationSize:int) -> None:
    with zipfile.ZipFile('./Experiments/'+f'{Dimension}_{PopulationSize}_Results.zip','w',zipfile.ZIP_DEFLATED) as file_zip:
        for variant in ['Base','FixedRandomSample','ProportionalRandomSample','Agglomerative','RandomParameters']:
            folder_variant = './Experiments/'+variant
            for root , _ , files in os.walk(folder_variant):
                for file in files:
                    file_path = os.path.join(root,file)

                    relative_file_path = os.path.relpath(file_path,start=os.path.dirname(folder_variant))
                    file_zip.write(file_path,relative_file_path)