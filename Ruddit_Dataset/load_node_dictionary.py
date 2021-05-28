import anytree
import numpy as np
from anytree import Node, RenderTree


def load_dictionary(node_dict):
	node_dictionary = np.load(node_dict, allow_pickle=True)
	node_dictionary = node_dictionary.item()
	return node_dictionary

def write_thread_structure_to_file(node_dictionary):
	with open('Thread_structure.txt', 'w') as output:
		for post, node_name in node_dictionary.items():	
			print('For Post ID: %s'%(post), file = output)	
			for pre, fill, node in RenderTree(node_name):
				print("%s%s" % (pre, node.name), file = output)
	output.close()

if __name__ == "__main__":

	node_dictionary = load_dictionary('node_dictionary.npy')
	write_thread_structure_to_file(node_dictionary)
