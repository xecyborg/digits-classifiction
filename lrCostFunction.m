function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

% Need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h_x = sigmoid(X * theta);
reg = (lambda/(2*m))*(sum(theta(2:end).^2));
J = (1/m) * (((-1*y') * log(h_x)) - ((1 - y') * log(1 - h_x))) + reg;

grad(1) = (1/m)*(X(:,1)'*(h_x - y));
grad(2:end) = (1/m)*(X(:,2:end)'*(h_x - y)) + (lambda/m)*theta(2:end);

grad = grad(:);

end
