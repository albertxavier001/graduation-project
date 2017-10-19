import os, sys, argparse, glob

root = '/Volumes/xavier/dataset/sintel2/'
out_folder = './split_scene'

all_scenes  = glob.glob(os.path.join(root, 'clean/*'))
for i, s in enumerate(all_scenes): 
	s = s.split('/')
	all_scenes[i] = s[-1]

training_scenes = []
test_scenes = []
for i in range(len(all_scenes)):
	for j in range(len(all_scenes)):
		if i >= j: continue
		training_list = list(all_scenes)
		training_list.remove(all_scenes[i])
		training_list.remove(all_scenes[j])
		test_list = [all_scenes[i], all_scenes[j]]
		training_scenes.append(sorted(training_list))
		test_scenes.append(sorted(test_list))


# for i in range(len(training_scenes)):
# 	print "training"
# 	print training_scenes[i]
# 	print "test"
# 	print test_scenes[i]

# print '# all_scenes', len(all_scenes)
# print '# training_scenes', len(training_scenes)

training_clean_str = ""
training_albedo_str = ""
training_shading_str = ""

test_clean_str = ""
test_albedo_str = ""
test_shading_str = ""

for i in range(len(training_scenes)):
	print i
	
	training_cleans = []
	training_albedos = []
	training_shadings = []

	test_cleans = []
	test_albedos = []
	test_shadings = []

	for scene in training_scenes[i]:
		training_cleans += glob.glob(os.path.join(root, 'clean', scene, "*.png"))
		training_albedos += glob.glob(os.path .join(root, 'albedo', scene, "*.png"))
		training_shadings += glob.glob(os.path .join(root, 'shading', scene, "*.png"))

	for scene in test_scenes[i]:
		test_cleans += glob.glob(os.path.join(root, 'clean', scene, "*.png"))
		test_albedos += glob.glob(os.path .join(root, 'albedo', scene, "*.png"))
		test_shadings += glob.glob(os.path .join(root, 'shading', scene, "*.png"))

	training_cleans = sorted(training_cleans)
	training_albedos = sorted(training_albedos)
	training_shadings = sorted(training_shadings)

	test_cleans = sorted(test_cleans)
	test_albedos = sorted(test_albedos)
	test_shadings = sorted(test_shadings)

	training_clean_str = '\n'.join(training_cleans)
	training_albedo_str = '\n'.join(training_albedos)
	training_shading_str = '\n'.join(training_shadings)

	test_clean_str = '\n'.join(test_cleans)
	test_albedo_str = '\n'.join(test_albedos)
	test_shading_str = '\n'.join(test_shadings)

	test_scenes[i] = sorted(test_scenes[i])
	
	split_scene_folder = os.path.join(out_folder, '{}.{}.as.test'.format(*test_scenes[i]))
	if not os.path.exists(split_scene_folder):
		os.makedirs(split_scene_folder)

	with open(os.path.join(split_scene_folder, 'training.clean.except.{}.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(training_clean_str)
	with open(os.path.join(split_scene_folder, 'training.albedo.except.{}.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(training_albedo_str)
	with open(os.path.join(split_scene_folder, 'training.shading.except.{}.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(training_shading_str)

	with open(os.path.join(split_scene_folder, 'test.clean.{}.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(test_clean_str)
	with open(os.path.join(split_scene_folder, 'test.albedo.{}.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(test_albedo_str)
	with open(os.path.join(split_scene_folder, 'test.shading.{}.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(test_shading_str)


