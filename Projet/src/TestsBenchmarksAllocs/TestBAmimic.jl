## Tests on matrices that look like matrices of BA Problems

@testset begin
    for k=1:50
        A1 = A_vect_collection[k]
        A1_full = VectToFull(A1)[1]
        A2 = B_collection[k]
        
        A = [A1_full A2]
        
        F = qr(A)
        F1 = qr(A1_full)
                
        Q1, R1, Q2, R2, Q1_mult_A2 = QR_PseudoBundle!(A1, A2)
        
        n1 = size(A1,2)
        
        k1 = size(Q1,1)
        k2 = size(Q2,1) #k1 ≥ k2
        
        Q = Q1*[I zeros(k1-k2, k2) ; zeros(k2, k1-k2) Q2]
        
        R = [R1  [Q1_mult_A2; R2]]
        
        @test norm(F.R - R[1:size(R,2), 1:size(R,2)]) ≤ 1e-13 #tests the unicity of the decomposition of A
        @test norm(F.Q - Q) ≤ 1e-13
        @test norm(A - Q*R) ≤ 1e-13 #tests if the QR decomposition of A is correct
    end
end