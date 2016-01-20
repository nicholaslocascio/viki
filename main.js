var Node = function(letter) {
	this.parent = null;
	this.children = [];
	this.current_str = "";
};

WORDS = ["hello", "wut"]

var Trie = function() {
	var base_node = Node(null, "", "");


	var add_word = function(word) {
		var node = base_node;
		while (word.length > 0) {
			var letter = word[-1];
			var new_child = Node(letter);
			node.add_child(new_child);
			word = word.substring(0, word.length-1);
			node = new_child;
		}
	};
	var add_node()

};