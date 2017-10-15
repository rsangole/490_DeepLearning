setwd("~/Documents/OneDrive/MSPA/490/Github")
library(readr)
library(dendextend)
library(tidyverse)

let <- read_csv('letters.csv')[,-1]
colnames(let) <- paste0('C',seq_len(81))
rownames(let) <- LETTERS
let <- as.matrix(let)
let
hclust(dist(let),method = 'ward.D2') -> let.clust
as.hclust(x = let.clust) %>% 
  as.dendrogram() %>%
  set("labels_col",k=8) %>% 
  set("labels_cex", 1.6) %>% plot(main='Ward-D2 H-Clustering on 81x1 letter arrays')
abline(h=5.6,lty=2,col='red')
let.clust$order

dataset <- read_csv("hidden.csv")
dataset %>% 
  tbl_df -> dataset
dataset <- reshape2::melt(dataset,'X1')
colnames(dataset) <- c('HiddenNode','Letter','Activation')
dataset$Letter <- factor(dataset$Letter,
                           levels = LETTERS[let.clust$order])
dataset$HiddenNode <- as.factor(dataset$HiddenNode)
dataset %>% 
  ggplot(aes(x=Letter,y=HiddenNode))+
  geom_tile(aes(fill=Activation))+
  geom_vline(xintercept = c(2.5,4.5,6.5,9.5,16.5,19.5,24.5),col='white')+
  theme_minimal()
