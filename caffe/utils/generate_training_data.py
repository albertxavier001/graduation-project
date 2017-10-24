import os, sys, argparse, glob, platform
from collections import OrderedDict


if platform.system() == 'Darwin': 
	root = '/Volumes/Transcend/dataset/sintel2/'
	dest_root_folder = './split_scene'
	caffe_root = 'home/lwp/workspace/direct-intrinsics/modified_caffe/caffe'
	pretrained_model = '/home/lwp/workspace/caffe_model/vgg16.caffemodel'
	template_root = '/Users/albertxavier/Box Sync/Graduation Project/graduation-project/caffe/utils/example_folder/template/'
elif platform.system() == 'Linux':
	root = '/home/lwp/workspace/sintel2'
	dest_root_folder = '/home/lwp/workspace/direct-intrinsics/training/split_scene'
	caffe_root = 'home/lwp/workspace/direct-intrinsics/modified_caffe/caffe'
	pretrained_model = '/home/lwp/workspace/caffe_model/vgg16.caffemodel'
	template_root = '/home/lwp/workspace/graduation-project/caffe/utils/example_folder/template'

all_scenes  = glob.glob(os.path.join(root, 'clean/*'))
for i, s in enumerate(all_scenes): 
	s = s.split('/')
	all_scenes[i] = s[-1]

training_scenes = []
test_scenes = []
for i in range(len(all_scenes)):
	training_list = list(all_scenes)
	training_list.remove(all_scenes[i])
	test_list = [all_scenes[i]]
	training_scenes.append(training_list)
	test_scenes.append(test_list)


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
	
	split_scene_folder = os.path.join(dest_root_folder, '{}'.format(*test_scenes[i]))
	if not os.path.exists(split_scene_folder): os.makedirs(split_scene_folder)


	""" set file """
	solver_file = os.path.join(split_scene_folder, 'solver.prototxt')
	prototxt_file = os.path.join(split_scene_folder, 'train_val.prototxt')
	snapshot_folder = os.path.join(split_scene_folder, 'snapshot', '')

	""" generate training lists """
	training_folder = 'training/'
	training_source = 'list/'
	if not os.path.exists(os.path.join(split_scene_folder, training_folder, training_source)): 
		os.makedirs(os.path.join(split_scene_folder, training_folder, training_source))
	
	training_clean_file = os.path.join(split_scene_folder, training_folder, training_source, 'training.clean.except.{}.txt'.format(*test_scenes[i]))
	training_albedo_file = os.path.join(split_scene_folder, training_folder, training_source, 'training.albedo.except.{}.txt'.format(*test_scenes[i]))
	training_shading_file = os.path.join(split_scene_folder, training_folder, training_source, 'training.shading.except.{}.txt'.format(*test_scenes[i]))
	with open(training_clean_file, 'w') as f:
		f.write(training_clean_str)
	with open(training_albedo_file, 'w') as f:
		f.write(training_albedo_str)
	with open(training_shading_file, 'w') as f:
		f.write(training_shading_str)

	""" generate solver """
	solver = OrderedDict([
		('net' , '"{}"'.format(prototxt_file)),
		('base_lr' , 0.002),
		('display' , 1),
		('average_loss' , 1),
		('max_iter' , 412500),
		('momentum' , 0.9),
		('weight_decay' , 1e-06),
		('snapshot' , 5000),
		('snapshot_prefix' , '"{}"'.format(snapshot_folder)),
		('lr_policy' , '"poly"'),
		('power' , 0.5),
		('iter_size' , 1),
		('share_blobs' , 'true')
	])
	
	with open(solver_file, 'w') as f:
		for (k,v) in solver.items():
			f.write(k + ': ' + str(v) + '\n')

	gpu = i % 3 + 1
	""" generate train.sh """
	if not os.path.exists(os.path.join(dest_root_folder, 'script', 'training')):
		os.makedirs(os.path.join(dest_root_folder, 'script', 'training'))
	with open(os.path.join(dest_root_folder, 'script', 'training', 'train_{}.sh'.format(*test_scenes[i])), 'w') as f:
		f.write('#!/usr/bin/env sh\n')
		f.write(os.path.join(caffe_root, 'build/tools/caffe') + ' train \\\n')
		f.write('-solver {} \\\n'.format(solver_file))
		f.write('-weights ' + pretrained_model + '\\\n')
		f.write('-gpu ' + str(gpu))

	""" generate training prototxt """
	prototxt = None
	with open(os.path.join(template_root, 'train_val.prototxt'), 'r') as f:
		prototxt = f.read()
		prototxt = prototxt.replace('@@@clean@@@', "'"+training_clean_file+"'")
		prototxt = prototxt.replace('@@@albedo@@@', "'"+training_albedo_file+"'")
		prototxt = prototxt.replace('@@@shading@@@', "'"+training_shading_file+"'")

	with open(prototxt_file, 'w') as f:
		f.write(prototxt)

	
	""" generate test lists """
	test_folder = 'test/'
	test_source = 'list/'
	if not os.path.exists(os.path.join(split_scene_folder, test_folder, test_source)): 
		os.makedirs(os.path.join(split_scene_folder, test_folder, test_source))

	with open(os.path.join(split_scene_folder, test_folder, test_source, 'test.clean.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(test_clean_str)
	with open(os.path.join(split_scene_folder, test_folder, test_source, 'test.albedo.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(test_albedo_str)
	with open(os.path.join(split_scene_folder, test_folder, test_source, 'test.shading.{}.txt'.format(*test_scenes[i])), 'w') as f:
		f.write(test_shading_str)


