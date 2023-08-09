import hail as hl

path = 'Bipolar'
bhr_phenotypes = hl.import_table('Dataset/Bipolar_disorder_customNA_pLoF_nvar493097_low0_high0.01_group1.txt')
bhr_phenotypes.describe()


# This finds all unique genes in table and converts into a list
a = list(bhr_phenotypes.aggregate(hl.agg.collect_as_set(bhr_phenotypes.gene)))

for gene in a:
    b = bhr_phenotypes.filter(bhr_phenotypes.gene==gene).select('locus', 'BETA', 'SE', 'AC', 'AF', )
    b.export('{}/Bipolar_{}.csv'.format(path, gene),delimiter='\t')

print("For debugging")