function run(vertices, max_iteration, delay){
    var startCost = cost(vertices);
    var currentTour = {vertices:vertices, cost:startCost};
    var bestTour = currentTour;
    var currentIteration = 0;

    function delayed_two_opt(){
        setTimeout(function(){
            if(currentIteration >= max_iteration){
                state_changed('Made '+currentIteration+' Iterations, started with cost: '+Math.floor(startCost)+', ended with cost: '+Math.floor(bestTour.cost), bestTour); // update canvas graph
                return;
            }
            currentIteration++;
            var newTour = two_opt(bestTour);
            state_changed('Running... Iteration: '+currentIteration+', Current: '+Math.floor(newTour.cost)+', Best: '+Math.floor(bestTour.cost), bestTour); // update canvas graph
            if(newTour.cost < bestTour.cost) bestTour = newTour;
            else max_iteration = currentIteration;
            delayed_two_opt();
        }, delay);
    }
    delayed_two_opt();
}

function two_opt(currentTour){
    var n = currentTour.vertices.length;
    var bestTour = currentTour;
    for(let i=1; i<n-2; i++){
        for(let j=i+1; j<n+1; j++){
            if(j-i == 1) continue;
            var swap = currentTour.vertices.slice(0,i).concat(currentTour.vertices.slice(i,j).reverse(), currentTour.vertices.slice(j,n))
            var newTour = {vertices:swap, cost:cost(swap)};
            if(newTour.cost < bestTour.cost) bestTour = newTour;
        }
    }
    return bestTour;
}

function cost(vertices){
    var total = 0;
    for(let i=1; i<vertices.length; i++){
        total += distance(vertices[i-1], vertices[i]);
    }
    total += distance(vertices[vertices.length-1], vertices[0]);
    return total;
}

function distance(vertice1, vertice2){ // replace with Manhattan, Haversine, etc...
    return Math.sqrt(Math.pow(vertice1[0]-vertice2[0], 2)+Math.pow(vertice1[1]-vertice2[1], 2));
}

function state_changed(message, event){}
