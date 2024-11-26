# Specify the path to your .sql file
file_path = 'queries.sql'

try:
    # Open and read the .sql file
    with open(file_path, 'r') as file:
        sql_content = file.read()

    # Print the content of the .sql file
    print(sql_content)
except FileNotFoundError:
    print(f"File not found: {file_path}")