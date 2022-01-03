function dCH = computeHomoGrad(x, UA, fem)

dCH = cell(3);
dxp = fem.penal * x.^(fem.penal - 1);
for i = 1:3
    for j = 1:3
        Ui = UA(:, i);
        Uj = UA(:, j);
        Uie = Ui(fem.edofs);
        Uje = Uj(fem.edofs);
        dCH{i, j} = sum(repmat(Uie, 1, 8) .* ...
            (dxp(:) * (fem.Ke1(:) - fem.Ke0(:))') .* ...
            kron(Uje, ones(1, 8)),2) / fem.num_elems;
    end
end