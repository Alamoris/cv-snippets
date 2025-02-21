import csv


def prepare_csv_reader(csv_path):
    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    r = csv_reader.__next__()  # Read headers
    return csv_file, csv_reader
