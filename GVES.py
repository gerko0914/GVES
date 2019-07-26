import argparse
import numpy as np
import pandas as pd
from pandas import DataFrame
from copy import deepcopy
from operator import itemgetter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf


def main():
	'''
	args = {'EXPRESSION_FILE','NETWORK_FILE','RESULT_FILE','SelectRatio',
			'VectorSize','numBiomarker'}
	'''
	args = parse_arguments()
	print(">>> Arguments")
	print(args)
	
	### 1. Load Data
	
	print(">>> Load Data")
	'''
	mRNA_data = DataFrame
		index :  OSEVENT + mRNA data and network data intersection genes 
		columns : whole samples
	edge = network list
		edge : mRNA data and network data intersection genes 
	'''
	mRNA_data, edge = load_mRNA(args.EXPRESSION_FILE, args.NETWORK_FILE)
	
	geneList = mRNA_data.index[1:]
	n_genes = len(geneList)
	gene2num, num2gene = make_geneDict(geneList)
	
	samples = mRNA_data.columns
	labels = mRNA_data.loc["OSEVNET",samples].values.astype(np.int)
	
	print(">>> Sample Clustering")
	sample_clustering = sampleClustering(mRNA_data,geneList,samples)
	n_cluster = len(sample_clustering.keys())
	
	good_sample, poor_sample = divide_sample(samples,labels)
	sam2num = make_sampleDict(samples)
	
	print("Good samples : ", len(good_sample))
	print("Poor samples : ", len(poor_sample))
	
	print(">>> Select Gene")
	ttest_list, select_gene = select_gene(sam2num,good_sample,poor_sample,mRNA_data,geneList,gene2num,edge,args.SelectRatio)
	
	new_geneList, new_edgeList = remake_gene_edge(select_gene,edge,num2gene)
	new_ngenes = len(new_geneList)
	new_gene2num, new_num2gene = make_geneDict(new_geneList)
	
	print(">>> Make network and pathway of each sample")
	geneCountDict = dict()
	for i, sam in enumerate(samples):
		if i % 10 == 0:
			print("Sample total, current sample : ", len(samples),i)
		adjMat = construct_adjMat(new_edgeList,ttest_list[i],new_ngenes)
		pathSet = order_generate_pathSet(adjMat)
		pathSet = list(pathSet)
		file_name = args.RESULT_FILE + sam + "_path.txt"
		print("Pathway of "+sam+"file name : "+file_name)
		fwrite_pathway(file_name, pathSet, new_num2gene)
		for path in pathSet:
			for gene in path:
				if gene in list(geneCountDict.keys()):
					geneCountDict[gene] += 1
				else:
					geneCountDict[gene] = 1
	
	count_sort = sorted(geneCountDict.items(),key=itemgetter(1),reverse=True)
	count_genes = np.array(count_sort)[:,0]
	
	print(">>> Make Sort gene2num num2gene")	
	sort_gene2num, sort_num2gene = make_geneDict(count_genes)	
	nonpathgene = list(set(select_gene) - set(count_genes))
	path_gene = len(count_genes)
	if len(nonpathgene) != 0:
		for i, gene in enumerate(nonpathgene)
			sort_gene2num[gene] = i+path_gene
			sort_num2gene[i+path_gene] = gene
	
	print(">>> Generate gene vector")
	for i, sam in enumerate(samples):
		path_file = args.RESULT_FILE + sam + "_path.txt"
		path_data = read_file(path_file)
		cbow_data = make_cbow_data(path_data,sort_gene2num,window_size)
		vector = CBOW(cbow_data,args.VectorSize,select_gene)
		file_name = args.RESULT_FILE + sam + "_vector.pkl"
		vector.to_pickle(file_name)
	
	gene_score = {}
	print(">>> Select Prognosis Gene")
	for cluster in range(n_cluster):
		cluster_sample = sampleClustering[cluster]
		print("Sample Length : ", len(cluster_sample))
		if len(cluster_sample) > 1:
			cluster_label = mRNA_data.loc["OSEVENT",cluster_sample].values.astype(np.int)
			for i in range(len(select_gene):
				if i % 500 == 0:
					print("Cluster, Gene : ", cluster,i)
				gene = select_gene[i]
				score = gene_clustering(cluster_sample,gene,args.RESULT_FILE,cluster_label)
				if gene not in gene_score.key():
					gene_score[gene] = score
				else:
					gene_score[gene] = gene_score[gene] + score
	sort_genescore = sorted(gene_score.items(),key =itemgetter(1), reverse = True)
	biomarker = sort_genescore[:,0]
	biomarker = biomarker[:args.numBiomarker]
	fwrite_biomarker(biomarker,args.RESULT_FILE)



def fwrite_biomarker(biomarkerList,filePath):
	with open(filePath+"biomarker.txt","w") as fout:
		for gene in biomarkerList:
			fout.write(gene)
			fout.write("\n")


def gene_clustering(sampleList, gene, filePath,RL):
	tmp_data = []
	for sam in sampleList:
		vector_name = filePath + sam + "_vector.pkl"
		df = pd.read_pickle(vector_name)
		column = df.columns
		tmp = df.loc[gene,column].values.astype(np.float32)
		tmp_data.append(tmp)
	kmeans = KMeans(n_clusters = 2,random_state=11).fit(tmp_data)
	PL = kmeans.labels_
	score = normalized_mutual_info_score(PL, RL,average_method='arithmetic')
	
	return score

	
def random_batch(data, size):
	random_inputs = []
	random_labels = []
	random_index = np.random.choice(range(len(data)), size, replace=False)
	# data = [ [[a,c], b], [[b,d], c] ]
	# random_input = [ [a,c], [b,d] ]
	# random_label = [ [b], [c] ]
	for i in random_index:
		random_inputs.append(data[i][0])  
		random_labels.append([data[i][1]]) 

	return random_inputs, random_labels		

	
def CBOW(data,vector_size,sel_gene):		
	training_epoch = 1000
	learning_rate = 0.01
	batch_size = 1000
	embedding_size = vector_size
	num_sampled = 800
	voc_size = len(sel_gene)
	
	inputs = tf.placeholder(tf.int32, shape=[batch_size,2])
	labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
	selected_embed = tf.nn.embedding_lookup(embeddings, inputs)
	embed_context = tf.reduce_mean(selected_embed,1)
	
	
	nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
	nce_biases = tf.Variable(tf.zeros([voc_size]))
	
	loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, embed_context, num_sampled, voc_size))
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		sess.run(tf.local_variables_initializer())
		
		loss_list = []
		before = 0
		current = 0
		for step in range(1, training_epoch + 1):
			batch_inputs, batch_labels = random_batch(data, batch_size)
			_, loss_val = sess.run([train_op, loss],
						feed_dict={inputs: batch_inputs,
							labels: batch_labels})
			loss_list.append(loss_val)
			
			if step % 100 == 0:
				current = np.mean(loss_list)
				if step != 100 and before < current:
					break
				else:
					before = current
			
				print("loss at step ", step, ": ", loss_val)
				print("Mean loss at step ",step, ": ", np.mean(loss_list))
				loss_list = []
		trained_embeddings = embeddings.eval()
	df_index = sel_gene	
	df_column = [i for i in range(trained_embeddings.shape[1])]
	vector_df = DataFrame(trained_embeddings,index = df_index, columns = df_column)
		
	return vector_df
	
	
def make_cbow_data(pathList,gene2num,window_size):
	cbow = []
	context_size = window_size * 2 + 1
	for data in pathList:
		for i in range(window_size,len(data)-window_size):
			target = gene2num[data[i]]
			context = [0] * context_size
			for j in range(1,window_size+1):
				context[window_size-j] = gene2num[data[i-j]]
				context[window_size+j] = gene2num[data[i+j]]
			del context[window_size]
			cbow.append([context,target])
	return cbow		

	
def read_file(dataFile):
	try:
		fin = open(dataFile)
	except:
		print("No file: %s" % dataFile)
		exit(1)
	else:	
		lines = list(map(lambda line:line.rstrip().split('\t'), fin.readlines()))
		fin.close()
	
	return lines	

	
def fwrite_pathway(file_name,pathList,new_num2gene):
	with open(file_name,"w") as fout:
		for path in pathList:
			if len(path) != 1:
				for gene in path:
					fout.write(new_num2gene[gene]+"\t")
				fout.write("\n")

				
def order_generate_pathSet(adjMat, maximumLength = 80, iterations=10):
	pathSet = list()
	n_genes = adjMat.shape[0]
	
	def generate_randomPath(src, adjMat, n_genes, maximumLength):
		path        = list()
		currentNode = src
		for step in range(maximumLength):
			path.append(currentNode)
			## adjMat = src * dest
			prob = deepcopy(adjMat[currentNode])
			## A walker dosen't go back to where he once went.
			prob[path] = 0.
			## A walker selects next node randomly.
			normalization = prob.sum()
			if normalization > 0.:
				prob /= normalization
				currentNode = np.random.choice(n_genes, size=1, p=prob)[0]
			else:
				## Random walking stops if a walker reaches a dead end.
				break
		return path
	
	for step in range(iterations):
		for src in range(n_genes):
			if src % 5000 == 0:
				print("iterations, n_genes, src:",step,n_genes,src)
			path = generate_randomPath(src, adjMat, n_genes, maximumLength)
			if path not in pathSet:
				pathSet.append(path)
	return pathSet

	
def construct_adjMat(edgeList, ttest_values, n_genes, gene2num,new_gene2num):

	adjMat  = np.zeros([n_genes, n_genes], dtype=np.float32)
	print(len(ttest_values))
	for edge in edgeList:
		src       = new_gene2num[edge[0]]
		dest      = new_gene2num[edge[1]]
		adjMat[src][dest] = ttest_values[gene2num[edge[1]]]
	return adjMat	
	

def remake_gene_edge(select_gene, original_edge,num2gene):
	new_gene = []
	for n_g in select_gene:
		new_gene.append(num2gene[n_g])
	print("Make New EdgeList")
	new_edge = []
	overlap_gene = set()
	for e in original_edge:
		g1, g2 = e[0],e[1]
		condition = ((g1 in new_gene) and (g2 in new_gene))
		if condition:
			new_edge.append([g1,g2])
			overlap_gene.add(g1)
			overlap_gene.add(g2)
	overlap_gene = list(overlap_gene)
	return overlap_gene, new_edge

	
def select_gene(sample2num,good,poor,data,gene,gene2num,edgeList,select_ratio):
	
	rank_list = np.zeros([len(all_sample),n_genes])
	ttest_list = np.zeros([len(all_sample),n_genes])
	n_genes = len(geneList)
	
	print("Good samples T-test")
	group_value = data.loc[gene,poor].values.astype(np.float32)
	for i,g_sam in enumerate(good):
		idx = sample2num[g_sam]
		if i%10 == 0:
			print("Good sample total, current sample : ", len(good),i)
		good_value = data.loc[gene,g_sam].values.astype(np.float32)
		rank_list[idx],ttest_list[idx] = gene_ttest(edgeList, good_value,group_value,n_genes,gene2num)
	
	print("Poor samples T-test")
	group_value = data.loc[gene,good].values.astype(np.float32)
	for i, p_sam in enumerate(poor):
		idx = sample2num[p_sam]
		if i%10 == 0:
			print("Poor sample total, current sample : ",len(poor),i)
		poor_value = data.loc[gene,p_sam].values.astype(np.float32)
		rank_list[idx],ttest_list[idx] = gene_ttest(edgeList,poor_value,group_value,n_genes,gene2num)
	
	rank_sum = sum(rank_list)
	gene_rank = (-rank_sum).argsort()
	gene_rank  = list(gene_rank)
	
	sel_gene = gene_rank[:int(n_genes*select_ratio)]
	
	return ttest_list,sel_gene

	
def gene_ttest(edgeList, one_value, group_value, n_genes, gene2num):
	
	statistic_list = []
	for i in range(n_genes):
		statistic = ttest_onesmaple(one_value[i], group_value[i])
		if math.isnan(statistic):
			statistic_list.append(0.)
		elif math.isinf(statistic):
			statistic_list.append(0.)
		else:
			statistic_list.append(abs(statistic))
	statistic_series = Series(statistic_list)
	rank = statistic_series.rank().values.astype(np.int)	
		
	return rank,statistic_list

	
def make_sampleDict(sampleList):
	for i, sam in enumerate(sampleList):
		sam2num[sam] = i
	return sam2num
	
	
def divide_sample(sampleList,labelList):
	good = []
	poor = []
	for la,sam in zip(labelList,sampleList):
		if la == 0:
			good.append(sam)
		else:
			poor.append(sam)
	return good,poor

	
def sampleClustering(data,genes,sampleList):
	
	train_expr = data.loc[genes,sampleList].values.astype(np.float32).T
	expr_zscored = np.zeros(data.shape)
	for i in range(train_expr.shape[1]):
		m = train_expr[:,i].mean()
		s = train_expr[:,i].std()
		if s > 0:
			expr_zscored[:,i] = (train_expr[:,i]-m)/s
	pca = PCA(n_components=2, random_state=11)
	expr_projected = pca.fit_transform(expr_zscored)		
	silhouetteScores = list()
	for i in [2,3,4]:
		kmeans = KMeans(n_clusters=i, random_state=11).fit(expr_projected)
		score  = silhouette_score(expr_projected, kmeans.labels_, random_state=11)
		silhouetteScores.append((i,score))
	silhouetteScores = sorted(silhouetteScores, key=itemgetter(1), reverse=True)
	n_clusters  = silhouetteScores[0][0]

	kmeans = KMeans(n_clusters=n_clusters, random_state=11).fit(expr_projected)
	labels = kmeans.labels_
	label2sample = {}
	
	for i in range(n_clusters):
		label2sample[i] = list()

	for label, sam in zip(labels, sampleList) :
		label2sample[label].append(sam)
	
	for key in label2sample.keys() :
		label2sample[key] = np.array(label2sample[key])
	
	return label2sample


def make_geneDict(geneList):
	gene2num = {}
	num2gene = {}
	for i, gene in enumerate(geneList):
		gene2num[gene] = i
		num2gene[i] = gene
	return gene2num,num2gene

	
def load_mRNA(expr_file,network_file) :
	fi_path = network_file
	with open(fi_path, "r") as fop :
		edges = []
		for line in fop :
			edges.append(line.strip().split("\t"))
	
	
	tmp_df = pd.read_pickle(expr_file)
	mRNA_genes = tmp_df.index
	edgeList = []
	overlap_genes = set()
	for edge in edges :
		g1, g2 = edge[0], edge[1]
		condition = ((g1 in mRNA_genes) and (g2 in mRNA_genes))
		if condition :
			edgeList.append([g1, g2])
			overlap_genes.add(g1)
			overlap_genes.add(g2)
	overlap_genes = list(overlap_genes)
	features = np.insert(overlap_genes,0, "OSEVENT")

	tmp_expr = tmp_df.ix[features[1:], tmp_df.columns].values.astype(np.float32)

	patientList = tmp_df.columns
	osevent = tmp_df.ix["OSEVENT", patientList].values.astype(np.int)
	table = np.vstack((osevent, tmp_expr))
	mRNA = DataFrame(table, index=features, columns=tmp_df.columns)
	return mRNA,edgeList

	
def parse_arguments():
	parser = argparse.Arguments()
	parser.add_argument('EXPRESSION_FILE', type=str)
	parser.add_argument('NETWORK_FILE', type=str)
	parser.add_argument('RESULT_FILE',type=str)
	parser.add_argument('-r', '--SelectRatio',type=int,default=20)
	parser.add_argument('-v', '--VectorSize',type=int,default=10)
	parser.add_argument('-n','--numBiomarker',type=int,default=150)
	return parser.parse_args()
	
if __name__=="__main__":
	main()