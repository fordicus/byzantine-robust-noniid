r"""Exp 3:
- Fix, e.g.,:
	- n=25, f=12
	- Number of iterations = 600
	- Not *Long tail* (alpha=1)
	- Always NonIID
	- Number of runs = 3
	- LR = 0.01
- Varies, e.g.,:
	- momentum=0, 0.9
	- ATK = LF, BF, Mimic, IPM, ALIE
	- 5 Aggregators: KRUM, CM, RFA, CClip, and A-CC
	- Bucketing or not
"""

from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
assert args.noniid
assert not args.LT

LOG_DIR = EXP_DIR + "exp3/"

if args.identifier: LOG_DIR += f"{args.identifier}/"
elif args.debug:	LOG_DIR += "debug/"
else:				LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "_output/"
LOG_DIR += (
	f"{args.agg}_{args.attack}_{args.momentum}"
	f"_s{args.bucketing:.2f}_df{args.dispersion_factor:.2f}"
	f"_vsz{args.val_set_sz}_gVm{args.beta_gV}_seed{args.seed}"
	f"_train_mb_sz{args.train_mb_sz}_tar{args.target}"
)


#---------------------------------------------------------------------------------
# 2025-01-20 (selective learning)
#.................................................................................

def BinaryToClassList(binaryInput, binaryLength=10):
	
	r"""
		Converts a binary string of a specified length to a list of integers (0 to binaryLength - 1)
		based on the positions of binary '1' in the input string.

		Args:
			binaryInput (str): A binary string (e.g., "0100010010").
			binaryLength (int): The expected length of the binary string. Default is 10.

		Returns:
			list: A list of integers corresponding to positions of '1' in the input.

		Raises:
			RuntimeError: If the input string's length does not match binaryLength.
	"""

	# Check if the input string length matches the expected binaryLength
	
	if len(binaryInput) != binaryLength:
		
		raise RuntimeError(
			f"Input string length must be {binaryLength}, but got {len(binaryInput)}."
		)

	# Initialize an empty list to store the output
	
	classList = []

	# Iterate through each character in the binary string
	
	for index, char in enumerate(binaryInput):
		
		# Check if the character is '1'
		
		if char == '1':
			
			# Append the index to the class list
			
			classList.append(index)
	
	return classList


def visualize_class_accuracy(
	class_accuracies,
	out_dir,
	max_batches_per_epoch,
	emphasized_classes,
	output_pdf_name = "exp3ca.pdf"):
		
	r"""
		Visualize class accuracy with emphasis on specific classes.
		
		Args:
			class_accuracies (list[dict]):	List of class accuracy dictionaries.
			out_dir (str):					Output directory for saving the plot.
			max_batches_per_epoch (int):	Number of batches per epoch.
			emphasized_classes (list[int]): List of class indices to emphasize.
	"""
	
	import os
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns

	if not os.path.exists(out_dir):
		
		os.makedirs(out_dir)

	# Convert to DataFrame
	
	class_accuracies_df = pd.DataFrame(class_accuracies)
	
	# Use Seaborn's color palette for emphasized classes
	
	palette = sns.color_palette("tab10", len(emphasized_classes))
	emphasized_colors = {cls: color for cls, color in zip(emphasized_classes, palette)}

	# Prepare the plot
	
	plt.figure(figsize=(10, 6))

	for digit in range(10):
		if digit in emphasized_classes:
			color = emphasized_colors[digit]
			linestyle = "-"
			alpha = 0.8
		else:
			color = "gray"
			linestyle = ":"
			alpha = 0.6

		plt.plot(
			class_accuracies_df["E"] * max_batches_per_epoch,
			class_accuracies_df[str(digit)],
			label=f"Class {digit}",
			color=color,
			linestyle=linestyle,
			alpha=alpha,
		)

	plt.xlabel("Iterations")
	plt.ylabel("Class Accuracy (%)")
	plt.title("Classwise Accuracy over Iterations")
	plt.legend(loc="lower right", title="Classes")
	plt.grid(True)

	# Save the plot
	output_path = os.path.join(out_dir, output_pdf_name)
	plt.savefig(output_path, bbox_inches="tight", dpi=720)
	plt.close()
	print(f"[saved] {output_path}")


#.................................................................................
# args.bucketing: ratio, e.g., 0.05 to be multiplied to MedAbsDiff
#.................................................................................

is_avg_no_byzantine  = ((args.agg == "avg") and (args.f == 0))
is_personalization	 = (args.target != "1111111111")
# is_dynamic_byzantine = ((args.client_delta > 0.0) and (args.f == 0))

if args.debug:
	
	MAX_BATCHES_PER_EPOCH = 1
	EPOCHS = 1
	
elif ((is_avg_no_byzantine) or (is_personalization)):
	
	# iterate less as personalizing
	
	EPOCHS = 5
	MAX_BATCHES_PER_EPOCH = 30

else:
	
	EPOCHS = 20
	MAX_BATCHES_PER_EPOCH = 30

print(
	f"\n"
	f"\t>>MAX_BATCHES_PER_EPOCH: {MAX_BATCHES_PER_EPOCH}\n"
	f"\t>>EPOCHS: {EPOCHS}")

#.................................................................................

_TARGET_CLASSES_ = BinaryToClassList(args.target, binaryLength=10)

#.................................................................................

if not args.plot:
	
	#-----------------------------------------------------------------------------
	# 2025-01-20 (selective learning)
	#.............................................................................
	
	main(
		args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH,
		target_classes = _TARGET_CLASSES_
	)
	
else:
	
	# Temporarily put the import functions here to avoid
	# random error stops the running processes.
	
	import os
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from codes.parser import extract_validation_entries

	# 5.5in is the text width of iclr2022 and 11 is the font size
	
	font = {"size": 10}
	plt.rc("font", **font)

	def exp_grid():
		
		switchNo = 2
		
		if switchNo == 0:		# Whole lots of experiments
			
			for agg in ["krum", "cm", "byro", "cp", "rfa"]:
			# for agg in ["byro", "cp"]:
			# for agg in ["byro"]:
				for seed in [0, 1, 2]:
				# for seed in [0]:
					# for bucketing in [0.0, 2.0]:
					# for bucketing in [2.0]:
					for bucketing in [0.0]:
						for momentum in [0.0, 0.9]:
						# for momentum in [0.0]:
						# for momentum in [0.9]:
							for attack in ["BF", "LF", "mimic", "IPM", "ALIE"]:
							# for attack in ["mimic", "IPM", "ALIE"]:
							# for attack in ["ALIE"]:
								yield agg, attack, momentum, bucketing, seed
								
		elif switchNo == 1:		# A-CC vs. ALIE 
								# It can also be used for personalization.
			
			for agg in ["byro"]:
				for seed in [0]:
					for bucketing in [0.0]:
						for momentum in [0.9]:
							for attack in ["ALIE"]:
								yield agg, attack, momentum, bucketing, seed
								
		elif switchNo == 2:		# Dynamic Byzantine: A-CC vs. CClip
			
			for agg in ["byro", "cp"]:
				for seed in [0, 1, 2]:
					for bucketing in [0.0]:
						for momentum in [0.0, 0.9]:
							for attack in ["ALIE"]:
								yield agg, attack, momentum, bucketing, seed
								
		elif switchNo == 1:		# A-CC vs. ALIE 
								# It can also be used for personalization.
			
			for agg in ["byro"]:
				for seed in [0]:
					for bucketing in [0.0]:
						for momentum in [0.9]:
							for attack in ["ALIE"]:
								yield agg, attack, momentum, bucketing, seed
								
		elif switchNo == -1:	# AVG with f = 0 (baseline of personalization)
			
			for agg in ["avg"]:
				for seed in [0]:
					for bucketing in [0.0]:
						for momentum in [0.9]:
							for attack in ["ALIE"]:
								yield agg, attack, momentum, bucketing, seed

	results = []
	class_accuracies = []
	counter = 1

	#-----------------------------------------------------------------------------
	# 2025-01-20 (selective learning)
	#.............................................................................
	
	for agg, attack, momentum, bucketing, seed in exp_grid():
		
		grid_identifier = (
			f"{agg}_{attack}_{momentum}_s{bucketing:.2f}_df{args.dispersion_factor:.2f}"
			f"_vsz{args.val_set_sz}_gVm{args.beta_gV}_seed{seed}"
			f"_train_mb_sz{args.train_mb_sz}_tar{args.target}"
		)
		path = INP_DIR + grid_identifier + "/stats"
		
		try:
			
			values, class_acc, counter = extract_validation_entries(path, counter)
			class_accuracies.extend(class_acc)
			
			for v in values:
				
				if	 agg == "cp":	str_agg = "CClip"
				elif agg == "byro": str_agg = "A-CC"
				else:				str_agg = agg.upper()
				
				results.append(
					{
						"Iterations":	v["E"] * MAX_BATCHES_PER_EPOCH,
						"Accuracy (%)":	v["top1"],
						"ATK":			attack,
						"AGG":			str_agg,
						r"$\beta$":		momentum,
						"seed":			seed,
						r"s":			str(bucketing),
					}
				)
				
		except Exception as e:
			
			pass
			
	
	# {"personalization", "byzantine", "both", "none"}
	
	# VISUAL_SUMMARY_SWITCH = "personalization"
	VISUAL_SUMMARY_SWITCH = "byzantine"
	# VISUAL_SUMMARY_SWITCH = "both"
	# VISUAL_SUMMARY_SWITCH = "none"
	
	print(f"\nVISUAL_SUMMARY_SWITCH: {VISUAL_SUMMARY_SWITCH}\n")
	
	if ((VISUAL_SUMMARY_SWITCH == "personalization") or 
		(VISUAL_SUMMARY_SWITCH == "both")):
		
		print(class_accuracies)
		
		visualize_class_accuracy(
			class_accuracies, 
			OUT_DIR, 
			MAX_BATCHES_PER_EPOCH, 
			emphasized_classes		= _TARGET_CLASSES_,
			output_pdf_name			= f"exp3ca-{args.target}.pdf"
		)

	if ((VISUAL_SUMMARY_SWITCH == "byzantine") or 
		(VISUAL_SUMMARY_SWITCH == "both")):

		results = pd.DataFrame(results)
		print(results)

		if not os.path.exists(OUT_DIR):
			
			os.makedirs(OUT_DIR)

		results.to_csv(OUT_DIR + "exp3.csv", index=None)
		
		print("")
		print("[saved] " + OUT_DIR + "exp3.csv")

		if args.client_delta == 0:	# the standard (fixed) Byzantine FL
			
			# sns.set(font_scale=1.7)
			g = sns.relplot(
				data	 = results,
				x		 = "Iterations",
				y		 = "Accuracy (%)",
				style	 = r"$\beta$",
				col		 = "ATK",
				row		 = "AGG",
				hue		 = r"s",							# for single seed
				height	 = 1.25,
				aspect	 = 2 / 1.25,
				palette	 = sns.color_palette("Set1", 2),	# for single seed
				# legend = False,
				# ci	 = None,
				kind	 = "line",
				# facet_kws = {"margin_titles": True},
			)
			g.set(xlim=(0, 600), ylim=(0, 100))
			g.set_axis_labels("Iterations", "Accuracy (%)")
			g.set_titles(row_template="{row_name}", col_template="{col_name}")
			g.fig.subplots_adjust(wspace=0.1)

			g.fig.savefig(OUT_DIR + "exp3.pdf", bbox_inches="tight", dpi=720)
			print("[saved] " + OUT_DIR + "exp3.pdf")
			
		else:
			
			FontSz = 18
			
			# Set global font settings
			plt.rcParams.update({
				"font.size":		FontSz,		# Set global font size
				"font.family":		"Arial",	# Set font to Arial
				"axes.labelsize":	FontSz,		# Axis labels font size
				"axes.titlesize":	FontSz,		# Title font size
				"legend.fontsize":	FontSz,		# Legend font size
				"xtick.labelsize":	FontSz,		# X-tick labels font size
				"ytick.labelsize":	FontSz,		# Y-tick labels font size
			})
			
			g = sns.relplot(
				data=results,
				x="Iterations",
				y="Accuracy (%)",
				style=r"$\beta$",
				col="AGG",		# Columns for A-CC and CClip
				row=None,		# No row distinction (horizontally stacked)
				hue=r"$\beta$",	# Use momentum (0.0, 0.9) for different line styles
				height=2.5,		# Adjust height
				aspect=1.5,		# Control width-to-height ratio
				palette=sns.color_palette("Set1", 1),	# Keep a single color across plots
				kind="line",
				dashes={0.0: "", 0.9: (4, 2)},			# Solid for 0.0, dashed for 0.9
			)

			# Set column titles to just "A-CC" and "CClip"
			g.set_titles(col_template="{col_name}")

			# Adjust axis labels
			g.set_axis_labels("Iterations", "Accuracy")

			# Set xticks for all subplots
			for ax in g.axes.flat:
				ax.set_xticks([0, 200, 400, 600])

			# Ensure only the leftmost subplot has a y-axis label
			for ax in g.axes.flat:
				ax.set_ylabel("Accuracy (%)")
			g.axes.flat[1].set_ylabel("")  # Remove y-label from second subplot

			# Ensure x-axis labels appear only once at the bottom
			for ax in g.axes.flat:
				ax.set_xlabel("Iterations")

			# Save the updated figure
			g.fig.subplots_adjust(wspace=0.1)  # Adjust horizontal spacing
			g.fig.savefig(OUT_DIR + "dynamic_byzantine_experiment.pdf", bbox_inches="tight", dpi=720)

			print("[saved] " + OUT_DIR + "dynamic_byzantine_experiment.pdf")
