from shared.document_utils import read_csv_file

def extract_health_data(csv_path):
    df = read_csv_file(csv_path)
    # autres traitements si nécessaire
    return df
