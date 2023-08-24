import csv

def check_csv_structure(file_path):
    problematic_rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row_number, row in enumerate(csv_reader, start=2):  # Start from row 2
            if len(row) != 2:
                problematic_rows.append((row_number, ",".join(row)))

    return problematic_rows

file_path = "data.csv"
problematic_rows = check_csv_structure(file_path)

if problematic_rows:
    print("Rows with more than two values:")
    for row_number, row_content in problematic_rows:
        print(f"Row {row_number}: {row_content}")
else:
    print("CSV file structure is consistent.")
