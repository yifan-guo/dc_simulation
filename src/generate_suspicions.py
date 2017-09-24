import pandas as pd
import numpy as np

data = pd.read_csv('data_extraction/output/individual_pair_data/adversary_disagree_with_majority.csv', sep='\t')
df = pd.DataFrame(data=data)

sLength = len(df['experiment'])
df = df.assign(guess = pd.Series(np.random.randn(sLength)).values)

thresholds = [0.3, 0.4, 0.5, 0.6]

#add another threshold (don't suspect until 15 sec, only when every neighbor has selected a color)
time_thresholds = [20]


# list_of_suspicions = []

for threshold in thresholds:
	for time_threshold in time_thresholds:
		for experiment in df['experiment'].unique():
			for game in df[df['experiment'].isin([experiment])]['game'].unique():
				# print('exp: ' + experiment + 'game: ' + str(game))
				indices = df[df['experiment'].isin([experiment]) & df['game'].isin([game])]['time'].index
				for idx in indices:
				# for tmp in df[df['experiment'].isin([experiment]) & df['game'].isin([game])]['time']:
					tmp = df[df['experiment'].isin([experiment]) & df['game'].isin([game])]['time'].ix[idx]
					disagree_times = tmp.strip('[')
					disagree_times = disagree_times.strip(']')
					disagree_times = [int(x) for x in disagree_times.split(' ')]
					acc = 0
					suspect = False
					for i in range(1, 60):
						if i in disagree_times:
							acc += 1
						if (acc / i) >= threshold and i > time_threshold:
							suspect = True
						if suspect:
							break
					df.set_value(idx, 'guess', suspect)
					# df[df['experiment'].isin([experiment]) & df['game'].isin([game]) & df['time'].isin[tmp]]['guess'] = suspect
					# list_of_suspicions.append(suspect)

		# list_of_suspicions = pd.Series(list_of_suspicions)
		# df = df.assign(guess = list_of_suspicions.values)
		df.to_csv('data_extraction/output/individual_pair_data/guesses.csv', sep='\t')

		# correct = 0
		# indices = df['adversary'].index
		# for idx in indices:
		# 	if int(df['adversary'].ix[idx]) == int(df['guess'].ix[idx]):
		# 		correct += 1
		# final_accuracy = float(correct) / float(len(df['adversary']))
		# print('accuracy: ' + str(final_accuracy))


		data2 = pd.read_csv('data_extraction/output/individual_pair_data/guesses.csv', sep='\t')
		df2 = pd.DataFrame(data=data2)


			#add another threshold (don't suspect until 15 sec, only when every neighbor has selected a color)

		correct = 0
		indices = df2['adversary'].index
		total_reg = 0
		total_adv = 0
		mislabel_regular = 0
		correctly_label_adversary = 0
		for idx in indices:
			actual = int(df2['adversary'].ix[idx])
			guess = int(df2['guess'].ix[idx])
			if actual == 0:
				#regular
				total_reg += 1
				if guess == 1:
					mislabel_regular += 1
			elif actual == 1:
				#adversary
				total_adv += 1
				if guess == 1:
					correctly_label_adversary += 1
		regulars_mislabeled = float(mislabel_regular) / float(total_reg)
		regulars_correctly_labeled = 1 - regulars_mislabeled
		adversaries_correctly_labeled = float(correctly_label_adversary) / float(total_adv)
		maximize_this = regulars_correctly_labeled + adversaries_correctly_labeled
		# print('regulars mislabeled as adversaries: ' + str(regulars_mislabeled))
		# print('adversaries guessed correctly: ' + str(adversaries_correctly_labeled))
		print("threshold: " + str(threshold) + " time_threshold: " + str(time_threshold) +  " regulars_correct: " + str(regulars_correctly_labeled) + " adversarries correct: " + str(adversaries_correctly_labeled) + " sum: " + str(maximize_this))
