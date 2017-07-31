
import os
import codecs

input_dir = 'text_original'
output_dir = 'text'

files = [f for f in os.listdir(input_dir) if '.txt' in f]


for f in files:

	print('processing:', f)

	with codecs.open(os.path.join(input_dir, f)) as inf:
		lines = inf.read().split('\n')

	found_start = False
	found_end = False
	new_lines = []

	for line in lines:
		if '*** START OF THIS PROJECT GUTENBERG EBOOK' in line:
			found_start = True
			continue
		elif line.strip() == 'THE END':
			found_end = True
			break

		if found_start:
			new_lines.append(line)

	# save to output
	with open(os.path.join(output_dir, f), 'w') as outf:
		outf.write('\n'.join(new_lines)+'\n')


