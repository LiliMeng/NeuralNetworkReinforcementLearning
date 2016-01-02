%learningRate comparison
figure
test = xlsread('data.xlsx', 'K1:M50');
x=[1:50];
l1=test(:,1);
l2=test(:,2);
l3=test(:,3);




plot(x(1:50),l1,x(1:50),l2,x(1:50),l3);
h_legend=legend('learningRate 0.1','learningRate 0.5','learningRate 0.8')
set(h_legend,'FontSize',15);

title('Different learning rates with 20 nodes in the hidden layer','FontSize', 15);
%scatter3(x,y,z,'filled')
xlabel('Epochs', 'FontSize', 15); % x-axis label
ylabel('Total Error', 'FontSize', 15);% y-axis label
