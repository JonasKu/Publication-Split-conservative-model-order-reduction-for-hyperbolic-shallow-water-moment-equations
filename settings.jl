__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # number spatial cells
    NCells::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    dx::Float64

    # time settings
    # end time
    tEnd::Float64;
    # time increment
    dt::Float64;
    # CFL number
    cfl::Float64;

    # degree PN
    N::Int64;

    # spatial grid
    x
    xMid

    # problem definitions
    problem::String;
    g::Float64
    nu::Float64
    lambda::Float64

    # DLRA settings
    r::Int

    function Settings(Nx::Int=2004;problem::String="shock")

        if problem == "shock" # water beam testcase
            cfl = 0.25;
            nu = 1.0;#10.0;#10*1e-2;
            lambda = 0.5;
            tEnd = 0.2
            r = 5
            N = 102;
        elseif problem == "sqrt" # waterbeam with intial sqrt downstream profile
            cfl = 0.25;
            nu = 1.0;#10.0;#10*1e-2;
            lambda = 0.5;
            tEnd = 0.2
            r = 5
            N = 102;
        elseif problem == "KowalskiTorrihon"
            # taken from Tabel1 in the paper:
            # Analysis and Numerical Simulation of Hyperbolic
            # Shallow Water Moment Equations
            # Julian Koellermeier and Marvin Rominger
            cfl = 0.25;
            nu = 0.1;#10.0;#10*1e-2;
            lambda = 0.1;
            tEnd = 0.2;
            r = 5
            N = 102;
        else
            cfl = 0.1;
            nu = 1e-5;
            lambda = 0.1;
            tEnd = 0.2;
            r = 10;
            N = 202;
        end

        # spatial grid setting
        NCells = Nx - 1;
        a = -1.0; # left boundary
        b = 1.0; # right boundary
        x = collect(range(a,stop = b,length = NCells));
        dx = x[2]-x[1];
        x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        xMid = x[1:(end-1)].+0.5*dx

        # time settings
        dt = cfl*dx;

        g = 9.81;

        # build class
        new(Nx,NCells,a,b,dx,tEnd,dt,cfl,N,x,xMid,problem,g,nu,lambda,r);
    end

end

function IC(obj::Settings,x)
    y = ones(size(x));
    if obj.problem == "cos"
        for j = 1:length(y);
            if x[j] > -0.5 && x[j] < 0.5
                y[j] = 0.1*(cos(2*pi*x[j])+1)+1.0
            end
        end
    elseif obj.problem == "TwoLayer"
        for j = 1:length(y);
            if x[j] > -0.5 && x[j] < 0.5
                y[j] = 0.01*(cos(2*pi*x[j])+1)+1.0
            end
        end
    elseif obj.problem == "KowalskiTorrihon"
        for j = 1:length(y);
            if x[j] > -1 && x[j] < 1
                y[j] = 1 + exp(3*cos(pi*(x[j]+0.5)))/exp(4)
            end
        end
    else
        for j = 1:length(y);
            #if x[j] < 0.0 && x[j] > -0.2
            #    y[j] = 1.0;
            #else
            #    y[j] = 0.3;
            #end
            #y[j] = 0.1 * exp(-x[j]^2/0.01) .+ 0.3
            y[j] = 0.7/2*(-tanh(50*(x[j] - 0.2)) + tanh(50*(x[j] + 0.0))) + 0.3# + tanh(x[j] + 0.1)
        end
    end

    return y;
end

function ICProfile(obj::Settings,x,eta)
    y = ones(length(x),length(eta));
    if obj.problem == "cos"
        for j = 1:length(x)
            for k = 1:length(eta)
                if x[j] > -0.5 && x[j] < 0.5
                    y[j] = 0.1*(cos(2*pi*x[j])+1)
                end
            end
        end
    else
        for j = 1:length(y);
            if x[j] < 0.0
                y[j] = 1.0;
            else
                y[j] = 0.3;
            end
        end
    end

    return y;
end
