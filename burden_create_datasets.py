import hail as hl

path = '30690NA_gene_effects_2'
bhr_phenotypes = hl.import_table('Dataset/dec_bhr_ms_variant_ss_400k_final_thin_withnullburden_30690NApLoF_nvar427542_low0_high1e-05_group1_2.txt')
bhr_phenotypes.describe()


# This finds all unique genes in table and converts into a list
a = list(bhr_phenotypes.aggregate(hl.agg.collect_as_set(bhr_phenotypes.gene)))

for gene in a:
    b = bhr_phenotypes.filter(bhr_phenotypes.gene==gene).select('locus', 'BETA', 'SE', 'AC', 'AF', )
    b.export('{}/30690NA_{}.csv'.format(path, gene),delimiter='\t')

print("For debugging")