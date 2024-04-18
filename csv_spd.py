import csv

def create_spd_file(row_number, values):
    filename = f"save_{row_number}.spd"
    with open(filename, 'w') as file:
        for i, value in enumerate(values, start=380):
            file.write(f"{i} {value}\n")

def process_csv(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row_number, row in enumerate(csv_reader, start=1):
            values = [float(value.replace(',', '.')) for value in row]
            create_spd_file(row_number, values)


csv_file = "../data_filtre.csv"
process_csv(csv_file)