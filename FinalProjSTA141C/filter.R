full<-read.csv("Phones_accelerometer.csv")
one_user<-full[full$User=='a',]
sub_one_user<-one_user[,c(4,5,6,9,10)]
sub_one_user_one_device<-sub_one_user[sub_one_user$Device=="nexus4_1",]
sub_one_user_one_device<-sub_one_user_one_device[,c(-4)]
sub_one_user_one_device<-sub_one_user_one_device[sub_one_user_one_device$gt!="null",]
temp<-sub_one_user_one_device
strings<-sort(unique(temp$gt))
colors=1:length(strings)
names(colors)=strings
temp$gt=colors[temp$gt]
temp<-temp[ rowSums(is.na(temp)) == 0, ]
sub_one_user_one_device<-temp
write.csv(sub_one_user_one_device,'filter.csv')