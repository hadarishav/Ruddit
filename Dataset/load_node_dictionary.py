import argparse
from typing import Dict

import numpy as np
from anytree import Node, RenderTree


def load_dictionary(node_dict: str) -> Dict[str, Node]:
	"""
	Loads the dictionary containing the thread structure from npy.

	:param node_dict: Path to the npy file containing the thread structure.
	:return: Dictionary with post id as key and anytree Node as value.
	"""
	node_dictionary = np.load(node_dict, allow_pickle=True)
	node_dictionary = node_dictionary.item()
	return node_dictionary


def write_thread_structure_to_file(node_dictionary: Dict[str, Node]) -> None:
	"""
	Write the dictionary to a txt file to visualize the threads.

	:param node_dictionary: Dictionary with post id as key and anytree Node as value.
	"""
	with open('Thread_structure.txt', 'w') as output:
		for post, node_name in node_dictionary.items():	
			print('For Post ID: %s' % (post), file=output)
			for pre, fill, node in RenderTree(node_name):
				print("%s%s" % (pre, node.name), file=output)
	output.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Create thread structures")
	parser.add_argument('--path_to_npy', help="Path to node dictionary npy", type=str)
	args = parser.parse_args()
	node_dictionary = load_dictionary(args.path_to_npy if not None else 'node_dictionary.npy')
	write_thread_structure_to_file(node_dictionary)
