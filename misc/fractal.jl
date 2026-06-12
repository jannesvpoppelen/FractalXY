using CairoMakie, DelimitedFiles, LinearAlgebra

cd(@__DIR__)

struct Triangle
    a::Int
    b::Int
    c::Int
    center::Int
end

mutable struct Fractal
    triangles::Vector{Triangle}
    vertices::Dict{Int, ComplexF64}
    neighbours::Dict{Int, Vector{Int}}
    max_index::Int
    N :: Int
end

function init_sierpinski()
    v1 = 0.0 + 0.0im
    v2 = 1.0 + 0.0im
    v3 = 0.5 + 0.5*sqrt(3.0)im
    center = (v1 + v2 + v3) / 3
    
    vertices = Dict(1 => v1, 2 => v2, 3 => v3, 4 => center)
    neighbours = Dict(1 => [4], 2 => [4], 3 => [4], 4 => [1, 2, 3])
    triangles = [Triangle(1, 2, 3, 4)]
    
    return Fractal(triangles, vertices, neighbours, 4, 4)
end

function add_vertex(frac::Fractal, pos::ComplexF64, tol=1e-10)
    for (idx, v) in frac.vertices
        abs(v - pos) < tol && return idx
    end
    
    frac.max_index += 1
    frac.vertices[frac.max_index] = pos
    frac.neighbours[frac.max_index] = Int[]
    return frac.max_index
end

function add_edge(frac::Fractal, i::Int, j::Int)
    j ∉ frac.neighbours[i] && push!(frac.neighbours[i], j)
    i ∉ frac.neighbours[j] && push!(frac.neighbours[j], i)
end

function inflate!(frac::Fractal)
    new_triangles = Triangle[]
    
    for tri in frac.triangles
        pa, pb, pc = frac.vertices[tri.a], frac.vertices[tri.b], frac.vertices[tri.c]
        
        m_ab, m_bc, m_ca = (pa + pb) / 2, (pb + pc) / 2, (pc + pa) / 2
        mid_ab = add_vertex(frac, m_ab)
        mid_bc = add_vertex(frac, m_bc)
        mid_ca = add_vertex(frac, m_ca)
        
        delete!(frac.vertices, tri.center)
        delete!(frac.neighbours, tri.center)
        for v in [tri.a, tri.b, tri.c]
            filter!(x -> x != tri.center, frac.neighbours[v])
        end
        
        center_a = (pa + m_ab + m_ca) / 3
        idx_ca = add_vertex(frac, center_a)
        add_edge(frac, tri.a, idx_ca)
        add_edge(frac, mid_ab, idx_ca)
        add_edge(frac, mid_ca, idx_ca)
        push!(new_triangles, Triangle(tri.a, mid_ab, mid_ca, idx_ca))
        
        center_b = (pb + m_bc + m_ab) / 3
        idx_cb = add_vertex(frac, center_b)
        add_edge(frac, tri.b, idx_cb)
        add_edge(frac, mid_bc, idx_cb)
        add_edge(frac, mid_ab, idx_cb)
        push!(new_triangles, Triangle(tri.b, mid_bc, mid_ab, idx_cb))
        
        center_c = (pc + m_ca + m_bc) / 3
        idx_cc = add_vertex(frac, center_c)
        add_edge(frac, tri.c, idx_cc)
        add_edge(frac, mid_ca, idx_cc)
        add_edge(frac, mid_bc, idx_cc)
        push!(new_triangles, Triangle(tri.c, mid_ca, mid_bc, idx_cc))
    end
    
    frac.triangles = new_triangles
    return frac
end

function generate_sierpinski(generation::Int)
    frac = init_sierpinski()
    for _ in 1:generation
        inflate!(frac)
    end
    renumber_vertices!(frac)
    frac.N = length(frac.vertices)
    return frac
end

function renumber_vertices!(frac::Fractal)
    old_indices = sort(collect(keys(frac.vertices)))
    new_map = Dict(old => new for (new, old) in enumerate(old_indices))
    
    new_vertices = Dict(new_map[old] => pos for (old, pos) in frac.vertices)
    new_neighbours = Dict(new_map[old] => [new_map[n] for n in neighbors] 
                         for (old, neighbors) in frac.neighbours)
    new_triangles = [Triangle(new_map[t.a], new_map[t.b], new_map[t.c], new_map[t.center]) 
                    for t in frac.triangles]
    
    frac.vertices = new_vertices
    frac.neighbours = new_neighbours
    frac.triangles = new_triangles
    frac.max_index = length(new_vertices)
end

function adjacency_matrix(frac::Fractal)
    n = length(frac.vertices)
    indices = sort(collect(keys(frac.vertices)))
    idx_map = Dict(idx => i for (i, idx) in enumerate(indices))
    
    A = zeros(Int, n, n)
    for (v, neighbors) in frac.neighbours
        i = idx_map[v]
        for u in neighbors
            A[i, idx_map[u]] = 1
        end
    end
    return A
end

function get_positions(frac::Fractal)
    indices = sort(collect(keys(frac.vertices)))
    return [frac.vertices[idx] for idx in indices], indices
end

function get_edge_list(frac::Fractal)
    edges = Tuple{Int,Int}[]
    for (v, neighbors) in frac.neighbours
        for u in neighbors
            if v < u
                push!(edges, (v-1, u-1)) # 0-based indexing
            end
        end
    end
    return sort(edges)
end

function plot_sierpinski(frac::Fractal; show_indices=false)
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    positions, indices = get_positions(frac)
    
    # Shade triangles
    for tri in frac.triangles
        pa, pb, pc = frac.vertices[tri.a], frac.vertices[tri.b], frac.vertices[tri.c]
        xs = [real(pa), real(pb), real(pc)]
        ys = [imag(pa), imag(pb), imag(pc)]
        poly!(ax, Point2f.(xs, ys), color=(:gray, 0.2))
    end
    
    # Draw edges
    for (v, neighbors) in frac.neighbours
        pv = frac.vertices[v]
        for u in neighbors
            pu = frac.vertices[u]
            lines!(ax, [real(pv), real(pu)], [imag(pv), imag(pu)], color=:black, linewidth=2)
        end
    end
    
    scatter!(ax, real.(positions), imag.(positions), color=:black, markersize=13)
    
    if show_indices
        for (i, pos) in enumerate(positions)
            text!(ax, real(pos), imag(pos), text=string(indices[i]), fontsize=25, align=(:left, :center))
        end
    end

    hidedecorations!(ax)
    hidespines!(ax)

    fig

end

frac0 = generate_sierpinski(0);
frac1 = generate_sierpinski(1);
frac2 = generate_sierpinski(2);
frac3 = generate_sierpinski(3);
frac4 = generate_sierpinski(4);
frac5 = generate_sierpinski(5);
frac6 = generate_sierpinski(6);

display(plot_sierpinski(frac0, show_indices=false))
display(plot_sierpinski(frac1, show_indices=true))
display(plot_sierpinski(frac2, show_indices=false))
display(plot_sierpinski(frac3, show_indices=false))
display(plot_sierpinski(frac4, show_indices=false))
display(plot_sierpinski(frac5, show_indices=false))
display(plot_sierpinski(frac6, show_indices=false))

save("frac.pdf", plot_sierpinski(frac2, show_indices=false))

e1 = get_edge_list(frac1)
e2 = get_edge_list(frac2)
e3 = get_edge_list(frac3)
e4 = get_edge_list(frac4)

writedlm("edges1.txt", e1)
writedlm("edges2.txt", e2)
writedlm("edges3.txt", e3)
writedlm("edges4.txt", e4)

v1, _ = get_positions(frac1)
v2, _ = get_positions(frac2)
v3, _ = get_positions(frac3)
v4, _ = get_positions(frac4)

writedlm("vertices1.txt", hcat(real.(v1), imag.(v1)))
writedlm("vertices2.txt", hcat(real.(v2), imag.(v2)))
writedlm("vertices3.txt", hcat(real.(v3), imag.(v3)))
writedlm("vertices4.txt", hcat(real.(v4), imag.(v4)))

