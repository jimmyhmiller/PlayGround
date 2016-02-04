/*
csp.go(function* () {
  var table = csp.chan();

  csp.go(player, ["ping", table]);
  csp.go(player, ["pong", table]);

  yield csp.put(table, {hits: 0});
  yield csp.timeout(1000);
  table.close();
});


function* player(name, table) {
  while (true) {
    var ball = yield csp.take(table);
    if (ball === csp.CLOSED) {
      console.log(name + ": table's gone");
      return;
    }
    ball.hits += 1;
    console.log(name + " " + ball.hits);
    yield csp.timeout(100);
    yield csp.put(table, ball);
  }
}
*/


let go = macro {
    rule {
        $fn($args (,) ...);
    } => {
        csp.go($fn, [$args (,) ...]);
    }
}


macro (->) {
    rule infix {
        $value:expr | $chan;
    } => {
        yield csp.put($chan, $value);
    }
}


macro (<-) {
    rule infix {
        $var | $chan:expr;
    } => {
        var $var = yield csp.take($chan);
    }
}

macro time {
    rule {
        .sleep($milli);
    } => {
        yield csp.timeout($milli);
    }
}


var Ball = () => ({ hits: 0 })

function* main() {
    var table = csp.chan();
    go player("ping", table);
    go player("pong", table);

    Ball() -> table;
    time.sleep(1000);
    ball <- table;
}

function* player(name, table) {
    while (true) {
        ball <- table;
        ball.hits++;
        console.log(name + " " + ball.hits);
        time.sleep(100);
        ball -> table;
    }
}

