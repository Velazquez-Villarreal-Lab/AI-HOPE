library(stringr)
#step 1 merge sample and pt level metadata. 

# pt_df = read.csv( 
#     file = "data/TCGA_COAD/data_clinical_patient.txt",
#     sep = "\t",
#     skip = 4,
#     row.names =1,
#     header = T,
#     na.strings=c("","NA"),
#     stringsAsFactors = FALSE
#     ) 

# print(head(pt_df))

# sample_df = read.csv( 
#     file = "data/TCGA_COAD/data_clinical_sample.txt",
#     sep = "\t",
#     skip = 4,
#     row.names =1,
#     header = T,
#     na.strings=c("","NA"),
#     stringsAsFactors = FALSE
#     )
# print(head(sample_df))
# r_list = intersect(rownames(pt_df),rownames(sample_df) )

# out_df = cbind(pt_df[r_list,], sample_df[r_list,])
# out_df$PATIENT_ID = rownames(out_df)
# rownames(out_df) = make.names(out_df$SAMPLE_ID)
# ### rename key attributes
# colnames(out_df)[21] = "TUMOR_STAGE"

# # out_df <- ifelse(is.na(out_df), "none", out_df)
# out_df[is.na(out_df)] <- "none"

# print(head(out_df))

# out_df$OS_STATUS <- ifelse(out_df$OS_STATUS == "0:LIVING", 0, ifelse(out_df$OS_STATUS == "1:DECEASED", 1, out_df$OS_STATUS))
# out_df$PFS_STATUS <- ifelse(out_df$PFS_STATUS == "0:CENSORED", 0, ifelse(out_df$PFS_STATUS == "1:PROGRESSION", 1, out_df$PFS_STATUS))

# write.table( 
#     out_df,
#     file = "data/TCGA_COAD/pt_metadata.tsv" ,
#     quote = F,
#     sep = "\t",
#     row.names = T
#   )



### Step 2 merge treatment dara  

# treatment_df = read.csv( 
#     file = "data/TCGA_COAD/data_timeline_treatment.txt",
#     sep = "\t",
#     na.strings=c("","NA"),
#     header = T
# )
# sample_df = read.csv( 
#     file = "data/TCGA_COAD/pt_metadata.tsv",
#     sep = "\t",
#     row.names =1,
#     header = T,
#     na.strings=c("","NA"),
#     stringsAsFactors = FALSE
#     )
# # print(head(sample_df["PATIENT_ID"]))
# out_df = sample_df
# out_df$TREATMENT_TYPE = rep("none", nrow(out_df))
# out_df$AGENT = rep("none", nrow(out_df))
# out_df$MEASURE_OF_RESPONSE = rep("none", nrow(out_df))
# for( rn in rownames(sample_df) ){
#     pid = sample_df[ rn ,"PATIENT_ID"] 
#     if(dim(treatment_df[which(treatment_df$PATIENT_ID == pid), ])[1] >0){
#         print(dim(treatment_df[which(treatment_df$PATIENT_ID == pid), ])[1])
#         trt_df =  treatment_df[which(treatment_df$PATIENT_ID == pid), ]
#         TREATMENT_TYPE_list = c()
#         Agent_list = c()
#         Response_list = c()
#         print(unique(treatment_df[which(treatment_df$PATIENT_ID == pid),"TREATMENT_TYPE" ]))
#         for(t_id in unique(treatment_df[which(treatment_df$PATIENT_ID == pid),"TREATMENT_TYPE" ])){
#             if(!trimws(t_id) %in%  TREATMENT_TYPE_list){
#                 TREATMENT_TYPE_list = append(TREATMENT_TYPE_list, trimws(t_id) )
#             }

#         }
#         out_df[rn, "TREATMENT_TYPE"] = paste(sort(TREATMENT_TYPE_list), collapse = "| ")
#         print(paste(sort(TREATMENT_TYPE_list), collapse = "| "))

#         for(a_id in unique(treatment_df[which(treatment_df$PATIENT_ID == pid),"AGENT" ] )){
#             tmp_list = unlist(str_split( a_id , "\\+") )
#             print(tmp_list)
#             for(t_id in tmp_list){
#                 print(t_id)
#                 if(!trimws(t_id) %in% Agent_list) {
#                     Agent_list = append( Agent_list, trimws(t_id) )
#                 }
#             }
#         }
#         out_df[rn,"AGENT"] = paste(sort(Agent_list), collapse = "| ")
#         print(paste(sort(Agent_list), collapse = "| "))

#         for(a_id in unique(treatment_df[which(treatment_df$PATIENT_ID == pid),"MEASURE_OF_RESPONSE" ] )){
#             tmp_list = unlist(str_split( a_id , "\\+") )
#             print(tmp_list)
#             for(t_id in tmp_list){
#                 print(t_id)
#                 if(!trimws(t_id) %in% Response_list) {
#                     Response_list = append( Response_list, trimws(t_id) )
#                 }
#             }
#         }
#         out_df[rn,"MEASURE_OF_RESPONSE"] = paste(sort(Response_list), collapse = "| ")
#         print(paste(sort(Response_list), collapse = "| "))

        
#     }
    

# }

# print(head(out_df))

# write.table( 
#     out_df,
#     file = "data/TCGA_COAD/pt_trt_metadata.tsv" ,
#     quote = F,
#     sep = "\t",
#     row.names = T
#   )


# # Step 3 : merge mutation 
# # 
mutgene_df = read.csv( 
    file = "data/TCGA_COAD/Mutated_Genes.txt",
    sep = "\t",
    row.names =1,
    header = T
)


mutgene_df = mutgene_df[which(mutgene_df$Is.Cancer.Gene..source..OncoKB. == "Yes"),]
mutgene_df$Percentage <- as.numeric(gsub("%", "", mutgene_df$Freq)) 
mutgene_df = mutgene_df[order(mutgene_df$Percentage, decreasing = TRUE),]
mutgene_df = mutgene_df[which(mutgene_df$Percentage > 10),]
print(rownames(mutgene_df) )

sample_df = read.csv( 
    file = "data/TCGA_COAD/pt_trt_metadata.tsv",
    sep = "\t",
    row.names =1,
    na.strings=c("","NA"),
    header = T
    )

xena_df = read.csv( 
    file = "data/TCGA_COAD/mc3.v0.2.8.PUBLIC.toil.xena.tsv",
    sep = "\t",
    header = T,
    na.strings=c("","NA"),
    stringsAsFactors = FALSE
    ) 
xena_df[is.na(xena_df)] <- "none"
xena_df$SIFT <- ifelse(trimws(xena_df$SIFT) == "", "Unknown", xena_df$SIFT)

xena_df$sample = make.names(xena_df$sample)
print(head(xena_df) )


for(gn in rownames(mutgene_df) ){
    sample_df[,paste0(gn,"_mutation_status")]= rep(0, nrow(sample_df))
    sample_df[,paste0(gn,"_mutation_effect")]= rep("none", nrow(sample_df))
    sample_df[,paste0(gn,"_mutation_Amino_Acid_Change")]= rep("none", nrow(sample_df))
    sample_df[,paste0(gn,"_mutation_SIFT")]= rep("none", nrow(sample_df))
}


for(pt in rownames(sample_df) ){
    print(pt)
    
    for(gn in rownames(mutgene_df) ){
        if(nrow(xena_df[which(xena_df$sample==pt & xena_df$gene == gn ),] ) >0){
            gn_df = xena_df[which(xena_df$sample==pt & xena_df$gene == gn ),]
            print(gn_df)
            sample_df[pt,paste0(gn,"_mutation_status")] = 1
            sample_df[pt,paste0(gn,"_mutation_effect")] = paste(gn_df$effect, collapse = "| ")
            sample_df[pt,paste0(gn,"_mutation_Amino_Acid_Change")] = paste(gn_df$Amino_Acid_Change, collapse = "| ")
            gn_df$SIFT = str_split_fixed(gn_df$SIFT, "\\(", 2)[, 1]
            print(paste(gn_df$SIFT, collapse = "| "))
            c_list = trimws(c(gn_df$SIFT))
            print(c_list)
            


            if("tolerated" %in% c_list){
                sample_df[pt,paste0(gn,"_mutation_SIFT")] = "tolerated"
                
            } 
            if("deleterious_low_confidence" %in% c_list){
                sample_df[pt,paste0(gn,"_mutation_SIFT")] = "deleterious_low_confidence"
                
            } 

            if("deleterious" %in% c_list){
                sample_df[pt,paste0(gn,"_mutation_SIFT")] = "deleterious"
              
            } 
           
            print(sample_df[pt,paste0(gn,"_mutation_SIFT")])
        }
    
    }
}

    write.table( 
    sample_df,
    file = "data/TCGA_COAD/pt_mut_metadata.tsv" ,
    quote = F,
    sep = "\t",
    row.names = T
  )


# # ### Step 4 merge CNA 

cna_df = read.csv( 
    file = "data/TCGA_COAD/data_armlevel_cna.txt",
    sep = "\t",
    row.names =1,
    header = T,
    na.strings=c("","NA"),
    stringsAsFactors = FALSE
    )
print(tail(cna_df))
meta_df = read.csv( 
    file = "data/TCGA_COAD/pt_mut_metadata.tsv",
    sep = "\t",
    row.names =1,
    header = T,
    na.strings=c("","NA"),
    stringsAsFactors = FALSE
    )

cna_df[is.na(cna_df)] <- "none"
cna_df[cna_df == "Unchanged"] <- "Diploid"


for(i in 1:nrow(cna_df)){
    cna_df[i,"NAME"] = paste0("Chr",trimws(cna_df[i,"NAME"]), "_CNV_status")
}
rownames(cna_df) = cna_df$NAME
# print( paste( cna_df$NAME , collapse=",") )
cna_df= data.frame(t(cna_df))[-(1:2),]
# 

print(head(meta_df))
sid_list = intersect(rownames(cna_df) ,rownames(meta_df)  ) 
cna_df = cna_df[sid_list,]
meta_df = meta_df[sid_list,]
meta_df = cbind(meta_df,cna_df)
print(head(meta_df) )

write.table( 
    meta_df,
    file = "data/TCGA_COAD/pt_mut_CNV_metadata.tsv" ,
    quote = F,
    sep = "\t",
    row.names = T
  )

# # # # ### step 5 subset patient sample 


sample_df = read.csv( 
    file = "data/TCGA_COAD/pt_mut_CNV_metadata.tsv",
    sep = "\t",
    row.names =1,
    na.strings=c("","NA","none"),
    stringsAsFactors=FALSE,
    header = T
    )

print(head(sample_df))
xena_df = read.csv( 
    file = "data/TCGA_COAD/TCGA_GTEX_category.txt",
    sep = "\t",
    na.strings=c("","NA","none"),
    stringsAsFactors=FALSE,
    header = T
    ) 

# print(head(xena_df))
rownames(xena_df) = make.names(xena_df$sample)
id_list = intersect(rownames(xena_df),rownames(sample_df))
xena_df = xena_df[ id_list, ]
sample_df = sample_df[id_list,]


### final refine ####

### MSI_status ####

sample_df$MSI_status = rep("none", nrow(sample_df) )
for(i in 1:nrow(sample_df)){
    if(! is.na( sample_df[i,"MSI_SCORE_MANTIS"] ) ){
    if(sample_df[i,"MSI_SCORE_MANTIS"]  > 0.4){
        sample_df[i,"MSI_status"] = "MSI"
    }
    if(sample_df[i,"MSI_SCORE_MANTIS"] <= 0.4){
        sample_df[i,"MSI_status"] = "MSS"
    }
    }
}

### TMB_status ####

sample_df$TMB_status = rep("none", nrow(sample_df) )
for(i in 1:nrow(sample_df)){
    if(! is.na( sample_df[i,"TMB_NONSYNONYMOUS"] ) ){
    print(sample_df[i,"TMB_NONSYNONYMOUS"])
    if( sample_df[i,"TMB_NONSYNONYMOUS"] >=10.0){
        
        sample_df[i,"TMB_status"] = "Elevated"
    }
    if( sample_df[i,"TMB_NONSYNONYMOUS"] < 10.0){
        sample_df[i,"TMB_status"] = "Normal"
    }
    print(sample_df$TMB_status[i])
    }
}

for(i in 1:nrow(sample_df)){
    if(! is.na( sample_df[i,"TMB_NONSYNONYMOUS"] ) ){
    print(sample_df[i,"TMB_NONSYNONYMOUS"])
    if( sample_df[i,"TMB_NONSYNONYMOUS"] >=10.0){
        
        sample_df[i,"TMB_status"] = "Elevated"
    }
    if( sample_df[i,"TMB_NONSYNONYMOUS"] < 10.0){
        sample_df[i,"TMB_status"] = "Normal"
    }
    print(sample_df$TMB_status[i])
    }
}


    write.table( 
    sample_df,
    file = "data/TCGA_COAD/pt_xena_metadata.tsv" ,
    quote = F,
    sep = "\t",
    na = "none",
    row.names = T
  )
print(head(sample_df))
print(table(sample_df$KRAS_mutation_effect))