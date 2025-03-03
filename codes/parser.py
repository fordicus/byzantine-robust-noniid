import json, re

def convert_numeric_keys_to_strings(line):
	
	# Regex to match dictionary keys that are numbers and not enclosed in quotes
	
	return re.sub(r'(\{|,)\s*(\d+)\s*:', r'\1 "\2":', line)
	
def extract_validation_entries(path, counter, kw="validation"):
	
	r"""
		-----------------------------------------------------------------------------
		Load JSON file of `stats` and extract relevant validation data.
		
		Args:
		
			path (str):	   Path to the JSON file containing stats.
			counter (int): Counter for debugging and logging.
			kw (str):	   Key to filter entries, default is "validation".
			
		Returns:
		
			tuple: List of validation entries,
				   list of class accuracy data,
				   updated counter.
			
		.............................................................................
		2025-01-20 (selective learning): Aggregate classwise accuracy, `ClassAccuracy`
		.............................................................................
	"""
	
	# print(f"[{counter}] Reading json file {path}")
	
	validation = []
	class_accuracy = []

	try:
		
		with open(path, "r") as f:
			
			for line_no, line in enumerate(f, start=1):
				
				# print(f"Processing line {line_no}: {line.strip()}")  # Debugging each line
				
				line = line.strip().replace("'", '"')			# Replace single quotes with double quotes
				line = line.replace("nan", '"nan"')				# Replace nan with string
				line = convert_numeric_keys_to_strings(line)	# Convert (MNIST) numeric keys to strings
				
				try:
					
					data = json.loads(line)  # Attempt JSON parsing
					
				except json.JSONDecodeError as e:
					
					print(
						f"JSONDecodeError on line {line_no}: {e}\n"
						f"received line: {line}\n"
					)
					
					continue  # Skip problematic lines
				
				# Validate the data structure
				
				if "_meta" in data and data["_meta"].get("type") == kw:
					
					validation.append(data)
					
					if "ClassAccuracy" in data:
						
						# Convert numeric keys in ClassAccuracy to strings
						
						class_accuracy.append({
							"E": data["E"],
							**{str(k): v for k, v in data["ClassAccuracy"].items()}
						})
	
	except FileNotFoundError as e:
		
		print(f"File not found: {path}")
		
		raise e
		
	except Exception as e:
		
		print(f"Unexpected error while reading {path}: {e}")
		
		raise e

	counter += 1

	# Debugging outputs
	# print(f"Extracted {len(validation)} validation entries.")
	# print(f"Extracted {len(class_accuracy)} class accuracy entries.")
	
	return validation, class_accuracy, counter
