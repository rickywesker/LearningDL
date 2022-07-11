import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

    class Node(object):
        def __init__(self,layer_index,node_index):
            self.layer_index = layer_index
            self.node_index = node_index
            self.upstream = []
            self.downstream = []
            self.output = 0
            self.delta = 0
        
        def set_output(self,output):
            self.output = output
        
        def set_delta(self,delta):
            self.delta = delta

        def downstream_append(self,conn):
            self.downstream.append(conn)

        def upstream_append(self,conn):
            self.upstream.append(conn)
        
        def calc_output(self):
            output = reduce(lambda ret,conn:ret + conn.upstream_node.output * conn.weight,self.upstream,0)
            self.output = sigmoid(output)

        def calc_hidden_layer_delta(self):#hidden layer
            downstream_delta = reduce(lambda ret,conn:ret + conn.downstream_node.delta * conn.weight,self.downstream,0.0)
            self.delta = self.output * (1 - self.output) * downstream_delta

        def calc_output_layer_delta(self,label):
            self.delta = self.output * (1 - self.output) * (label - self.output)

        def __str__(self):
            node_str = "%u-%u:output:%f delta:%f" %(self.layer_index,self.node_index,self.output,self.delta)
            downstream_str = reduce(lambda ret,conn:ret + '\n\t' + str(conn),self.downstream,'')
            upstream_str = reduce(lambda ret,conn:ret + '\n\t' + str(conn),self.upstream,'')
            return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

    class ConstNode(object):
        def __init__(self,layer_index,node_index):
            self.layer_index = layer_index
            self.node_index = node_index
            self.downstream = []
            self.output = 1

        def downstream_append(self,conn):
            self.downstream.append(conn)
        
        def set_delta(self):
            downstream_delta = reduce(lambda ret,conn:ret + conn.downstream_node.delta * conn.weight,downstream_delta,0.0)
            self.delta = self.output * (1 - self.output) * downstream_delta
        
        def __str__(self):
            node_str = '%u-%u:output:1 delta:%f' % (self.layer_index,self.node_index,self.delta)
            downstream_str = reduce(lambda ret,conn:ret + '\n\t' + str(node),self.downstream,'')
            return node_str + '\n\tdownstream:' + downstream_str

    class Layer(object):
        def __init__(self,layer_index,node_count):
            self.layer_index = layer_index
            self.node_count = node_count
            self.nodes = [Node(layer_index,i) for i in range(self.node_count)]
            self.nodes.append(ConstNode(layer_index,self.node_count))

        def set_output(self,data):#input layer x1,x2,x3...
            for i in range(len(data)):
                self.nodes[i].set_output(data[i])
        
        def calc_output(self):
            for node in self.nodes[:-1]:
                node.calc_output()

        def dump(self):
            for node in self.nodes
                print(node)

    class Connection(object):
        def __init__(self,upstream_node,downstream_node):
            self.upstream_node = upstream_node
            self.downstream_node = downstream_node
            self.weight = random.uniform(-0.1,0.1)
            self.gradient = 0.0

        def calc_gradient(self):
            self.gradient = self.downstream_node.delta * self.upstream_node.output

        def get_gradient(self):
            return self.gradient

        def update_weight(self,rate):
            calc_gradient()
            self.weight += self.gradient * rate

        def __str__(self):
            return '(%u-%u) -> (%u-%u) = %f' % (
                self.upstream_node.layer_index,
                self.upstream_node.node_index,
                self.downstream_node.layer_index,
                self.downstream_node.node_index,
                self.weight
            )

    class Connections(object):
        def __init__(self):
            self.connections = []
        
        def connection_append(self,conn):
            self.connections.append(conn)

        def dump(self):
            for conn in self.connections:
                print(conn)

    class Network(object):
        def __init__(self,layers):#layers[i] <----> # of nodes in layer i
            self.connections = Connections()
            self.layers = []
            layer_count = len(layers)
            node_count = 0
            #how many layers?
            self.layers.extend(Layer(i,layers[i]) for i in range(layer_count))
            for layer in range(layer_count - 1):
                connections = [Connection(upstream_node,downstream_node)
                               for upstream_node in self.layers[layer].nodes
                               for downstream_node in self.layers[layer+1].nodes[:-1]]
                                    #cant get it?
                                    #temp = [Node(a,b)
                                    #for a in [1,2,3,4,5]
                                    #for b in [2,4,6,8]]  
                for conn in connections:
                    self.connections.add_connection(conn)
                    conn.downstream_node.upstream_append(conn)
                    conn.upstream_node.downstream_append(conn)
                    