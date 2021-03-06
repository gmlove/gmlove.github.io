(function(C){

Crafty.c("DiagonalLine", {
    init: function () {
        this.requires("2D, Canvas");
        this.bind("Draw", this._draw_me);
        this.ready = true;
    },
    color: function (color) {
        this.color = color;
        return this;
    },
    direction: function (direction) {
        this._direction = direction;
        return this;
    },
    width: function (width) {
        this.width = width;
        return this;
    },
    _draw_me: function (e) {
        var ctx = e.ctx;
        ctx.lineWidth = this.width;
        ctx.strokeStyle = this.color;
        ctx.beginPath();
        if (this._direction === 'down-up') {
            ctx.moveTo(e.pos._x, e.pos._y + e.pos._h);
            ctx.lineTo(e.pos._x + e.pos._w, e.pos._y);
        } else {
            ctx.moveTo(e.pos._x, e.pos._y);
            ctx.lineTo(e.pos._x + e.pos._w, e.pos._y + e.pos._h);
        }
        ctx.stroke();
    }
});

Crafty.c("Triangle", {
    init: function () {
        this.requires("2D, Canvas");
        this.bind("Draw", this._draw_me);
        this.ready = true;
    },
    color: function (color) {
        this.color = color;
        return this;
    },
    direction: function (direction) {
        this._direction = direction;
        return this;
    },
    width: function (width) {
        this.width = width;
        return this;
    },
    _draw_me: function (e) {
        var ctx = e.ctx;
        ctx.lineWidth = this.width;
        ctx.strokeStyle = this.color;
        ctx.beginPath();
        var points = [];
        if (this._direction === 't') {
            points = [[e.pos._x, e.pos._y + e.pos._h], [e.pos._x + e.pos._w / 2, e.pos._y], [e.pos._x + e.pos._w, e.pos._y + e.pos._h]];
        } else if (this._direction === 'r') {
            points = [[e.pos._x, e.pos._y], [e.pos._x + e.pos._w, e.pos._y + e.pos._h / 2], [e.pos._x, e.pos._y + e.pos._h]];
        } else if (this._direction === 'b') {
            points = [[e.pos._x, e.pos._y], [e.pos._x + e.pos._w / 2, e.pos._y + e.pos._h], [e.pos._x + e.pos._w, e.pos._y]];
        } else if (this._direction === 'l') {
            points = [[e.pos._x, e.pos._y + e.pos._h / 2], [e.pos._x + e.pos._w, e.pos._y], [e.pos._x + e.pos._w, e.pos._y + e.pos._h]];
        } else {
            throw new Error('unknown direction: ' + this._direction);
        }
        for (var i = 0; i < points.length; i++) {
            var point = points[i], pointNext = points[i + 1 === points.length ? 0 : i + 1];
            ctx.moveTo(point[0], point[1]);
            ctx.lineTo(pointNext[0], pointNext[1]);
        }
        ctx.stroke();
    }
});

function EventBus () {
    this.subscribers = {};
}
EventBus.prototype = {
    subscribe: function (eventID, callback) {
        var subscribers = this.subscribers[eventID] = this.subscribers[eventID] || [];
        subscribers.push(callback);
    },
    unsubscribe: function (eventID, callback) {
        var subscribers = this.subscribers[eventID] = this.subscribers[eventID] || [];
        var idx = subscribers.lastIndexOf(callback);
        if (idx !== -1) {
            subscribers.splice(idx, 1);
        }
    },
    emit: function (eventID, args) {
        var subscribers = this.subscribers[eventID] = this.subscribers[eventID] || [];
        for (var i = 0; i < subscribers.length; i++) {
            var subscriber = subscribers[i];
            subscriber(args);
        }
    }
}

function GameInteractor(router) {
    this.router = null;
}
GameInteractor.prototype = {
    start: function () {
        this.router.attachGridTable();
    }
}
function GameRouter(view, interactor, gridTableBuilder) {
    this.view = view;
    this.interactor = interactor;
    this.gridTableBuilder = gridTableBuilder;
}
GameRouter.prototype = {
    attachGridTable: function () {
        var gridTableRouter = this.gridTableBuilder.build(this.view);
        this.view.addView(gridTableRouter.view);
        gridTableRouter.interactor.start();
    }
}
function GameView() {
    this.width = 800, this.height = 700;
    Crafty.init(this.width, this.height, document.getElementById('game'));
}
GameView.prototype = {
    addView: function(view) {
    }
}
function GameBuilder() {
}
GameBuilder.prototype = {
    build: function () {
        var view = new GameView();
        var interactor = new GameInteractor();
        var router = new GameRouter(view, interactor, new GridTableBuilder());
        interactor.router = router;
        return router;
    }
}

var m = {
    argmax: function (arrOrObj) {
        if (!m.isArray(arrOrObj) && typeof(arrOrObj) === 'object') {
            var keys = Object.keys(arrOrObj);
        } else {
            var keys = m.range(arrOrObj.length);
        }

        var values = [];
        for (var i = 0; i < keys.length; i++) {
            values.push(arrOrObj[keys[i]]);
        }

        var idxOfMax = -1;
        for (var i = 0; i < values.length; i++) {
            if (idxOfMax === -1) {
                idxOfMax = i;
                continue;
            }
            if (values[idxOfMax] < values[i]) {
                idxOfMax = i;
            }
        }
        return keys[idxOfMax];
    },
    range: function (length) {
        var range = [];
        for (var i = 0; i < length; i++) {
            range.push(i);
        }
        return range;
    },
    randomInt: function (min, max) {
        min = min || 0, max = max || Number.MAX_SAFE_INTEGER;
        return Math.floor(Math.random() * (max - min) + min);
    },
    zeroArray: function (shape) {
        var arr = [];
        if (!shape.length) {
            return 0;
        }
        for (var i = 0; i < shape[0]; i++) {
            arr.push(m.zeroArray(shape.slice(1)));
        }
        return arr;
    },
    isArray: function (obj) {
        return !!(obj && obj.length !== undefined && typeof(obj) === 'object' && typeof(obj.splice) === 'function');
    },
    values: function (object) {
        var values = [];
        var keys = Object.keys(object);
        for (var i = 0; i < keys.length; i++) {
            values.push(object[keys[i]]);
        }
        // console.log( `values for object ${object}: ${values}`)
        return values;
    },
    eq: function (curr, exp, falseCallback) {
        if (typeof(curr) === 'number' && typeof(exp) === 'number' && Math.abs(curr - exp) < 1e-8) {
            return true;
        } else if (exp !== curr) {
            if (falseCallback) {
                falseCallback(curr, exp);
            }
            return false;
        }
        return true;
    },
    arrayEq: function (curr, exp, falseCallback) {
        if (curr.length !== exp.length) {
            if (falseCallback) {
                falseCallback(curr, exp);
            }
            return false;
        }
        if (!m.isArray(curr) || !m.isArray(exp)) {
            return m.eq(curr, exp, falseCallback);
        }
        if (curr.length) {
            if (m.isArray(curr[0])) {
                for (var i = 0; i < curr.length; i++) {
                    if(!m.arrayEq(curr[i], exp[i], falseCallback)) {
                        return false;
                    }
                }
            } else {
                for (var i = 0; i < curr.length; i++) {
                    if(!m.eq(curr[i], exp[i], falseCallback)) {
                        return false;
                    }
                }
            }
        }
        return true;
    },
    objEq: function(curr, exp, falseCallback) {
        var keysCurr = Object.keys(curr).sort(), keysExp = Object.keys(exp).sort();
        if (!m.arrayEq(keysCurr, keysExp, falseCallback)) {
            return false;
        }
        for (var i = 0; i < keysCurr.length; i++) {
            var key = keysCurr[i];
            if (m.isArray(curr[key]) && m.isArray(exp[key])) {
                if(!m.arrayEq(curr[key], exp[key], falseCallback)) {
                    return false;
                }
            } else {
                if(!m.eq(curr[key], exp[key], falseCallback)) {
                    return false;
                }
            }
        }
        return true;
    },
    assertTrue: function (expression) {
        if (expression !== true) {
            throw new Error('expression is not true, but ' + expression);
        }
    }
}
var config = {
    showValue: true
}

function GridCellQValue(t, r, b, l) {
    this.t = t, this.r = r, this.b = b, this.l = l;
}
GridCellQValue.prototype = {
    asArray: function () {
        return [this.t, this.r, this.b, this.l];
    }
}
function GridCell(type, opts) {
    opts = opts || {};
    this.type = type;
    this.value = opts.value || 0;
    this.qValue = opts.qValue || new GridCellQValue(0, 0, 0, 0);
    this._direction = opts.direction || undefined;
    this.reward = opts.reward || 0;
}

var CELL_TYPE = {
    ROOM: 'room',
    WALL: 'wall',
    SUCCESS: 'success',
    FAIL: 'fail'
}

function CellType(type) {
    this.type = type;
    m.assertTrue([CELL_TYPE.ROOM, CELL_TYPE.WALL, CELL_TYPE.SUCCESS, CELL_TYPE.FAIL].indexOf(type) != -1);
}

CellType.prototype = {
    isWall: function () {
        return this.type === CELL_TYPE.WALL;
    },
    isRoom: function () {
        return this.type === CELL_TYPE.ROOM;
    },
    isSuccess: function () {
        return this.type === CELL_TYPE.SUCCESS;
    },
    isFail: function () {
        return this.type === CELL_TYPE.FAIL;
    },
    isEnd: function () {
        return this.isSuccess() || this.isFail();
    }
}

CellType.room = new CellType(CELL_TYPE.ROOM)
CellType.wall = new CellType(CELL_TYPE.WALL)
CellType.success = new CellType(CELL_TYPE.SUCCESS)
CellType.fail = new CellType(CELL_TYPE.FAIL)

GridCell.prototype = {
    direction: function () {
        if (this.type.isWall()) {
            return null;
        }
        if (this._direction !== undefined) {
            return this._direction;
        }
        return ['t', 'r', 'b', 'l'][m.argmax(this.qValue.asArray())];
    },
    positiveReward: function () {
        return this.reward > 0;
    },
    currentValue: function () {
        if (this.type.isWall()) {
            return [];
        } else {
            return config.showValue ? [this.value] : this.qValue.asArray();
        }
    }
}
function GridTable(cells) {
    this.cells = cells;
    this.shape = {w: cells.length, h: cells[0].length};
}
GridTable.prototype = {
    cellAt: function (row, column) {
        if (row < 0 || row >= this.cells.length) {
            throw new Error('invalid row: ' + row)
        }
        if (column < 0 || column >= this.cells[0].length) {
            throw new Error('invalid column: ' + column)
        }
        return this.cells[row][column];
    },
    updateValues: function (states, values) {
        for (var i = 0; i < states.length; i++) {
            var state = states[i];
            var stateSplit = state.split('-');
            var row = parseInt(stateSplit[0]), column = parseInt(stateSplit[1]);
            var cell = this.cellAt(row, column);
            cell.value = values[i];
        }
    },
    updateQValues: function (states, qValuesAll) {
        for (var i = 0; i < states.length; i++) {
            var state = states[i];
            var stateSplit = state.split('-');
            var row = parseInt(stateSplit[0]), column = parseInt(stateSplit[1]);
            var cell = this.cellAt(row, column);
            var qValues = qValuesAll[i];
            if (Object.keys(qValues).length) {
                cell.qValue = new GridCellQValue(qValues.t, qValues.r, qValues.b, qValues.l);
            }
        }
    },
    updatePolicy: function (states, policy) {
        for (var i = 0; i < states.length; i++) {
            var state = states[i];
            var stateSplit = state.split('-');
            var row = parseInt(stateSplit[0]), column = parseInt(stateSplit[1]);
            var cell = this.cellAt(row, column);
            cell._direction = policy[i];
        }
    }
}
function GridTableInteractor(presenter, gridTable) {
    this.router = null;
    this.presenter = presenter; // {initTable, updateTable, startValueIteration, startPolicyIteration, finishIteration}
    this.gridTable = gridTable;
    this.mdp = this._mdp();
    this.solverVI = new MDPVISolver(this.mdp);
    this.solverPI = new MDPPISolver(this.mdp);
}
GridTableInteractor.prototype = {
    states: function () {
        var states = [];
        for (var i = 0; i < this.gridTable.cells.length; i++) {
            var cells = this.gridTable.cells[i];
            for (var j = 0; j < cells.length; j++) {
                var cell = cells[j];
                states.push(i + '-' + j);
            }
        }
        return states;
    },
    start: function () {
        this.presenter.initTable(this.gridTable);
        this.setupVISolver();
        this.setupPISolver();
    },
    setupVISolver: function () {
        var mdp = this.mdp, solver = this.solverVI, gridTable = this.gridTable, self = this;
        this.presenter.startValueIteration().subscribe(function () {
            var values = m.zeroArray([mdp.states.length]);
            var valuesNew = values;
            var iterations = 0;
            var interval = setInterval(function () {
                var qValuesAll = [];
                values = valuesNew;
                valuesNew = [];
                for (var i = 0; i < mdp.states.length; i++) {
                    var state = mdp.states[i];
                    var qValues = solver.qValues(values, state);
                    qValuesAll.push(qValues);
                    var value = solver.value(qValues);
                    valuesNew.push(value);
                }
                gridTable.updateValues(mdp.states, valuesNew);
                gridTable.updateQValues(mdp.states, qValuesAll);
                self.presenter.updateTable(gridTable);
                self.presenter.updateStatus('finished iteration ' + (++iterations));
                if (solver.converged(values, valuesNew)) {
                    clearInterval(interval);
                    self.presenter.updateStatus('solved!');
                    self.presenter.finishIteration();
                    var states = mdp.states;
                    var policy = {
                        nextAction: function (state) {
                            return m.argmax(qValuesAll[states.indexOf(state)]);
                        }
                    };
                }
            }, 3000);
        });
    },
    setupPISolver: function () {
        var mdp = this.mdp, solver = this.solverPI, gridTable = this.gridTable, self = this;
        this.presenter.startPolicyIteration().subscribe(function () {
            function solveForPolicy (policy, onSolved) {
                var policyValues = m.zeroArray([solver.states.length]);
                var policyValuesNew = policyValues;
                var iteration = 0;
                var interval = setInterval(function () {
                    policyValues = policyValuesNew;
                    policyValuesNew = solver.policyValues(policyValues, policy);

                    gridTable.updateValues(mdp.states, policyValuesNew);
                    self.presenter.updateTable(gridTable);
                    self.presenter.updateStatus('solve for policy iteration ' + (++iteration));
                    // console.log('policyValuesNew: ', policyValuesNew);
                    if (iteration === 1000) {
                        clearInterval(interval);
                        throw new Error('cannot converge after 1000 steps!');
                    }
                    if (solver.valueConverged(policyValues, policyValuesNew)) {
                        clearInterval(interval);
                        self.presenter.updateStatus('value converged for policy iteration!');
                        onSolved(policyValues);
                    }
                }, 100);
            }

            var policy = solver.randomPolicy();
            var policyNew = policy;
            var iterations = 0;
            gridTable.updatePolicy(mdp.states, policy);
            self.presenter.updateTable(gridTable);

            function solve () {
                policy = policyNew;
                solveForPolicy(policy, function (policyValues) {
                    policyNew = solver.improvePolicy(policyValues, policy);

                    gridTable.updateValues(mdp.states, policyValues);
                    gridTable.updatePolicy(mdp.states, policyNew);
                    self.presenter.updateTable(gridTable);
                    self.presenter.updateStatus('finished policy iteration ' + (++iterations));

                    if (solver.converged(policy, policyNew)) {
                        self.presenter.updateStatus('solved!');
                        self.presenter.finishIteration();

                        var states = mdp.states;
                        return {
                            nextAction: function (state) {
                                return policy[states.indexOf(state)];
                            }
                        };
                    }
                    setTimeout(solve, 3000);
                });

            }
            setTimeout(solve, 3000);
        });
    },
    _mdp: function () {
        var gridTable = this.gridTable;
        return {
            states: this.states(),
            availableActions: function (state) {
                var stateSplit = state.split('-');
                var x = parseInt(stateSplit[0]), y = parseInt(stateSplit[1]);
                var cell = gridTable.cellAt(x, y);
                if (cell.type.isRoom()) {
                    return ['t', 'l', 'r', 'b'];
                }
                return [];
            },
            reward: function (state, action, targetState) {
                var stateSplit = targetState.split('-');
                var x = parseInt(stateSplit[0]), y = parseInt(stateSplit[1]);
                var cell = gridTable.cellAt(x, y);
                return cell.reward;
            },
            transitions: function (state, action) {
                var stateSplit = state.split('-');
                var row = parseInt(stateSplit[0]), column = parseInt(stateSplit[1]);
                var cell = gridTable.cellAt(row, column);
                if (cell.type.isEnd() || cell.type.isWall()) {
                    throw new Error('Reward and wall cell cannot transition to other cell!');
                }
                var availableStates = { self: [row, column].join('-') }, transitions = {};
                if (row - 1 >= 0 && !gridTable.cellAt(row - 1, column).type.isWall()) {
                    availableStates.t = [row - 1, column].join('-');
                }
                if (column + 1 < gridTable.shape.h && !gridTable.cellAt(row, column + 1).type.isWall()) {
                    availableStates.r = [row, column + 1].join('-');
                }
                if (row + 1 < gridTable.shape.w && !gridTable.cellAt(row + 1, column).type.isWall()) {
                    availableStates.b = [row + 1, column].join('-');
                }
                if (column - 1 >= 0 && !gridTable.cellAt(row, column - 1).type.isWall()) {
                    availableStates.l = [row, column - 1].join('-');
                }

                var availableDirections = Object.keys(availableStates);
                if (!availableStates[action]) {
                    action = 'self';
                }
                transitions[availableStates[action]] = 0.7;
                var prob = (1 - transitions[availableStates[action]]) / (availableDirections.length - 1);
                for (var i = 0; i < availableDirections.length; i++) {
                    var direction = availableDirections[i];
                    if (direction !== action) {
                        transitions[availableStates[direction]] = prob;
                    }
                }
                return transitions;
            }
        };
    }
}
function GridCellView(cell, pos, wh) {
    this.pos = pos, this.wh = wh;
    var color = { 'room': '#228B22', 'wall': 'gray', 'success': '#DAA520', 'fail': '#B22222' };
    var type = cell.type.type;
    C.e('2D, Canvas, Color').attr({x: pos[0], y: pos[1], w: wh[0], h: wh[1]}).color(color[type]);
    var currentValue = cell.currentValue();
    var size = this.textSize = wh[0] / 7;
    if (currentValue.length === 1) {
        this.updateValue(currentValue);
    } else if (currentValue.length === 4) {
        this.updateQValue(currentValue);
    }

    this.updateDirection(cell);
}
GridCellView.prototype = {
    destroyQValueElements: function () {
        this.destroyElements(['upDownLine', 'downUpLine', 'qValueTextL', 'qValueTextB', 'qValueTextR', 'qValueTextT']);
    },
    updateQValue: function (currentValue) {
        var pos = this.pos, wh = this.wh, size = this.textSize;
        this.currentValue = currentValue;
        this.upDownLine = this.upDownLine || C.e('DiagonalLine').attr({x: pos[0], y: pos[1], w: wh[0], h: wh[1]}).color('white');
        this.downUpLine = this.downUpLine || C.e('DiagonalLine').attr({x: pos[0], y: pos[1], w: wh[0], h: wh[1]}).color('white').direction('down-up');
        this.qValueTextT = this.qValueTextT || C.e('2D, Canvas, Text').attr({x: pos[0] + wh[0] / 3, y: pos[1] + size}).textFont({size: size + 'px'});
        this.qValueTextR = this.qValueTextR || C.e('2D, Canvas, Text').attr({x: pos[0] + wh[0] / 3 * 2, y: pos[1] + wh[1] / 2 - size / 2}).textFont({size: size + 'px'});
        this.qValueTextB = this.qValueTextB || C.e('2D, Canvas, Text').attr({x: pos[0] + wh[0] / 3, y: pos[1] + wh[1] - size - size}).textFont({size: size + 'px'});
        this.qValueTextL = this.qValueTextL || C.e('2D, Canvas, Text').attr({x: pos[0] + size / 2, y: pos[1] + wh[1] / 2 - size / 2}).textFont({size: size + 'px'});
        var colors = ['white', 'white', 'white', 'white'];
        var policyDirection = m.argmax(currentValue);
        colors[policyDirection] = '#FF8C00';
        this.updateText(this.qValueTextT, currentValue[0].toFixed(2), colors[0]);
        this.updateText(this.qValueTextR, currentValue[1].toFixed(2), colors[1]);
        this.updateText(this.qValueTextB, currentValue[2].toFixed(2), colors[2]);
        this.updateText(this.qValueTextL, currentValue[3].toFixed(2), colors[3]);
    },
    updateText: function (element, text, color) {
        var textOld = element.text();
        if (textOld !== text) {
            element.text(text).textColor(color);
            if (textOld) {
                this.blink(element, color);
            }
        }
    },
    blink: function (element, originalColor, blinkColor, timeout) {
        blinkColor = blinkColor || '#FF8C00', timeout = timeout || 500;
        element.textColor(blinkColor);
        setTimeout(function() {
            element.textColor(originalColor);
        }, timeout);
    },
    destroyValueElements: function () {
        this.destroyElements(['valueText']);
    },
    updateValue: function (currentValue) {
        var pos = this.pos, wh = this.wh, size = this.textSize;
        this.valueText = this.valueText || C.e('2D, Canvas, Text').attr({x: pos[0] + wh[0] / 3, y: pos[1] + wh[1] / 2 - size / 2}).textColor('white').textFont({size: size + 'px'});
        this.updateText(this.valueText, currentValue[0].toFixed(2), 'white');
    },
    destroyElements: function (elements) {
        for (var i = 0; i < elements.length; i++) {
            if (this[elements[i]]) {
                this[elements[i]].destroy();
                this[elements[i]] = null;
            }
        }
    },
    updateDirection: function (cell) {
        var pos = this.pos, wh = this.wh;
        var direction = cell.direction();
        var directionSize = 5;
        var x = -1, y = -1;
        if (direction === 't') {
            var x = pos[0] + (wh[0] - directionSize) / 2, y = pos[1] + 2;
        } else if (direction === 'r') {
            var x = pos[0] + wh[0] - directionSize - 2, y = pos[1] + (wh[1] - directionSize) / 2;
        } else if (direction === 'b') {
            var x = pos[0] + (wh[0] - directionSize) / 2, y = pos[1] + wh[1] - directionSize - 2;
        } else if (direction === 'l') {
            var x = pos[0] + 2, y = pos[1] + (wh[1] - directionSize) / 2;
        }
        if (x !== -1 && y !== -1) {
            if (!this.direction) {
                this.direction = C.e('Triangle').attr({x: x, y: y, w: directionSize, h: directionSize}).direction(direction).color('red');
            } else {
                this.direction.attr({x: x, y: y, w: directionSize, h: directionSize}).direction(direction);
            }
        }
    },
    update: function (cell) {
        var currentValue = cell.currentValue();
        if (currentValue.length === 1) {
            this.destroyQValueElements();
            this.updateValue(currentValue);
        } else if (currentValue.length === 4) {
            this.destroyValueElements();
            this.updateQValue(currentValue);
        }
        this.updateDirection(cell);
    }
}
function TextButton (viewVector, margin, text, textSize, onClick) {
    this.disabled = false;
    var x = viewVector.x, y = viewVector.y, w = viewVector.w, h = viewVector.h;
    var self = this;
    var btn = this.btn = Crafty.e('2D, Canvas, Color, Mouse').attr(viewVector).color('#228B22')
        .bind('MouseOver', function(MouseEvent){ if (self.disabled) return; self.btn.color('#178117'); })
        .bind('MouseOut', function(MouseEvent){ if (self.disabled) return; self.btn.color('#228B22'); })
        .bind('MouseDown', function(MouseEvent){ if (self.disabled) return; self.btn.color('#0B760B'); })
        .bind('MouseUp', function(MouseEvent){ if (self.disabled) return; self.btn.color('#178117'); })
        .bind('Click', function(MouseEvent){
            if (self.disabled) return;
            onClick();
        });
    this.text = C.e('2D, Canvas, Text').attr({x: x + margin / 2, y: y + h / 4}).text(text).textColor('white').textFont({size: textSize + 'px'});
}
TextButton.prototype = {
    disable: function () {
        this.btn.color('gray');
        this.disabled = true;
    },
    enable: function () {
        this.btn.color('#228B22');
        this.disabled = false;
    }
}
function ControllerView (gridView) {
    var vector = gridView.controllerVector;
    var margin = 20, textSize = 18;

    var x = vector.x, y = margin + vector.y;
    var w = Math.min(140, vector.w - margin);
    var h = Math.min(w / 3, vector.h - margin);
    var self = this;
    this.solverVIBtn = new TextButton({x: x, y: y, w: w, h: h}, margin, 'Value Iteration', textSize, function () {
        gridView.onStartValueIteration();
        self.disableSolveBtns();
    });

    x = x + w + margin * 2;
    this.solverPIBtn = new TextButton({x: x, y: y, w: w, h: h}, margin, 'Policy Iteration', textSize, function () {
        gridView.onStartPolicyIteration();
        self.disableSolveBtns();
    });

    x = x + w + margin * 2;
    this.toggleQValueBtn = new TextButton({x: x, y: y, w: w, h: h}, margin, 'Toggle Q Value', textSize, function () {
        config.showValue = !config.showValue;
        gridView.updateTable();
    });
}
ControllerView.prototype = {
    enableSolveBtns: function () {
        this.solverPIBtn.enable();
        this.solverVIBtn.enable();
    },
    disableSolveBtns: function () {
        this.solverPIBtn.disable();
        this.solverVIBtn.disable();
    }
}
function GridTableView(parentView) {
    this.parentView = parentView;
    this.tableVector = {x: 0, y: 0, w: parentView.width, h: parentView.height * 0.8};
    this.controllerVector = {x: 0, y: parentView.height * 0.8, w: parentView.width, h: parentView.height * 0.15};
    this.controllerView = null;
    this.cellMargin = 2;
    this.borderWidth = 0;
    this.cellViews = [];
    this.statusView = C.e('2D, Canvas, Text').attr({x: 0, y: parentView.height * 0.95, w: parentView.width, h: parentView.height * 0.05}).textColor('gray').textFont({size: parentView.height * 0.03 + 'px'});
    this.eventBus = new EventBus();
}
GridTableView.prototype = {
    cellWH: function (cellNumW, cellNumH) {
        return [
            (this.tableVector.w - 2 * this.borderWidth - this.cellMargin) / cellNumW - this.cellMargin,
            (this.tableVector.h - 2 * this.borderWidth - this.cellMargin) / cellNumH - this.cellMargin
        ]
    },
    cellPos: function (x, y, wh) {
        var cellW = wh[0], cellH = wh[1];
        return [
            this.borderWidth + this.cellMargin + (this.cellMargin + cellW) * x,
            this.borderWidth + this.cellMargin + (this.cellMargin + cellH) * y
        ];
    },
    updateStatus: function (text) {
        console.log(text);
        this.statusView.text(text);
    },
    initTable: function(gridTable) {
        this.gridTable = gridTable;
        var cellWH = this.cellWH(gridTable.cells[0].length, gridTable.cells.length);
        for (var h = 0; h < gridTable.cells.length; h++) {
            var cells = gridTable.cells[h];
            var cellViews = [];
            this.cellViews.push(cellViews);
            for (var w = 0; w < cells.length; w++) {
                cellViews.push(new GridCellView(cells[w], this.cellPos(w, h, cellWH), cellWH));
            }
        }
        this.controllerView = new ControllerView(this);
    },
    updateTable: function(gridTable) {
        gridTable = this.gridTable = gridTable || this.gridTable;
        if (!gridTable) {
            return;
        }
        for (var h = 0; h < gridTable.cells.length; h++) {
            var cells = gridTable.cells[h];
            var cellViews = this.cellViews[h];
            for (var w = 0; w < cells.length; w++) {
                cellViews[w].update(cells[w]);
            }
        }
    },
    onStartValueIteration: function () {
        this.eventBus.emit('value-iteration');
    },
    startValueIteration: function () {
        var self = this;
        return {
            subscribe: function (callback) {
                self.eventBus.subscribe('value-iteration', callback);
            }
        };
    },
    onStartPolicyIteration: function () {
        this.eventBus.emit('policy-iteration');
    },
    startPolicyIteration: function () {
        var self = this;
        return {
            subscribe: function (callback) {
                self.eventBus.subscribe('policy-iteration', callback);
            }
        };
    },
    finishIteration: function () {
        this.controllerView.enableSolveBtns();
    }
}
function GridTableRouter(view, interactor) {
    this.view = view;
    this.interactor = interactor;
}
function GridTableBuilder() {
}
GridTableBuilder.prototype = {
    build: function (parentView) {
        var view = new GridTableView(parentView);
        var c = function (type, arg1, arg2) {
            if (type.isEnd()) {
                return new GridCell(type, { reward: arg1 });
            } else if (type.isRoom()) {
                return new GridCell(type, { value: arg1, qValue: arg2 });
            } else {
                return new GridCell(type);
            }
        }
        var gridTable = new GridTable([
            [c(CellType.room), c(CellType.room), c(CellType.room), c(CellType.room), c(CellType.success, 1)],
            [c(CellType.room), c(CellType.wall), c(CellType.wall), c(CellType.room), c(CellType.fail, -1)],
            [c(CellType.room), c(CellType.room), c(CellType.room), c(CellType.room), c(CellType.room)]
        ]);
        var interactor = new GridTableInteractor(view, gridTable);
        var router = new GridTableRouter(view, interactor);
        interactor.router = router;
        return router;
    }
}


function MDP() {
    this.states = [];
}
MDP.prototype = {
    availableActions: function (state) {
    },
    reward: function (state, action, targetState) {
    },
    transitions: function (state, action) {
    }
}

function MDPPolicy () {
}
MDPPolicy.prototype = {
    nextAction: function (state) {}
}

function MDPVISolver (mdp) {
    this.mdp = mdp;
    this.states = this.mdp.states;
    this.gama = 0.5;
}

MDPVISolver.prototype = {
    solve: function() {
        var values = m.zeroArray([this.states.length]);
        var valuesNew = values;
        var iterations = 0;
        do {
            var qValuesAll = [];
            values = valuesNew;
            valuesNew = [];
            for (var i = 0; i < this.states.length; i++) {
                var state = this.states[i];
                var qValues = this.qValues(values, state);
                qValuesAll.push(qValues);
                var value = this.value(qValues);
                valuesNew.push(value);
            }
            console.log('finished iteration ' + (++iterations));
            // console.log('values: ', values);
        } while(!this.converged(values, valuesNew));
        var states = this.states;
        var policy = {
            nextAction: function (state) {
                return m.argmax(qValuesAll[states.indexOf(state)]);
            }
        };
        return policy;
    },
    converged: function (valuesOld, valuesCurr) {
        return m.arrayEq(valuesOld, valuesCurr);
    },
    value: function (qValues) {
        qValues = m.values(qValues);
        if (!qValues.length) {
            return 0;
        }
        return Math.max.apply(null, qValues);
    },
    qValues: function (valuesOld, state) {
        var availableActions = this.mdp.availableActions(state);
        var qValues = {};
        for (var j = 0; j < availableActions.length; j++) {
            var availableAction = availableActions[j];
            var qValue = this.qValue(valuesOld, state, availableAction)
            qValues[availableAction] = qValue;
        }
        // console.log(`qValues: state ${state}, qValues ${qValues}`)
        return qValues;
    },
    qValue: function (valuesOld, state, action) {
        var qValue = 0;
        var transitions = this.mdp.transitions(state, action);
        var targetStates = Object.keys(transitions);
        for (var i = 0; i < targetStates.length; i++) {
            var targetState = targetStates[i];
            var value = valuesOld[this.states.indexOf(targetState)];
            qValue += transitions[targetState] * (this.mdp.reward(state, action, targetState) + this.gama * value);
        }
        // console.log(`qValue: state ${state}, action ${action}, qValue ${qValue}`)
        return qValue;
    }
}

function MDPPISolver (mdp) {
    this.mdp = mdp;
    this.states = this.mdp.states;
    this.gama = 0.5;
}

MDPPISolver.prototype = {
    solve: function () {
        var policy = this.randomPolicy();
        var policyNew = policy;
        do {
            policy = policyNew;
            values = this.solveForPolicy(policy);
            policyNew = this.improvePolicy(values, policy);
        } while (!this.converged(policy, policyNew));
        var states = this.states;
        return {
            nextAction: function (state) {
                return policyNew[states.indexOf(state)];
            }
        };
    },
    improvePolicy: function (values, policy) {
        var policyNew = [];
        for (var i = 0; i < this.states.length; i++) {
            var state = this.states[i];
            var betterAction = policy[i];
            var maxQValue = 0;
            var availableActions = this.mdp.availableActions(state);
            for (var j = 0; j < availableActions.length; j++) {
                var availableAction = availableActions[j];
                var qValue = this.qValue(values, state, availableAction);
                if (maxQValue < qValue) {
                    maxQValue = qValue;
                    betterAction = availableAction;
                }
            }
            policyNew.push(betterAction);
        }
        return policyNew;
    },
    solveForPolicy: function (policy) {
        var policyValues = m.zeroArray([this.states.length]);
        var policyValuesNew = policyValues;
        var iteration = 0;
        do {
            policyValues = policyValuesNew;
            policyValuesNew = this.policyValues(policyValues, policy);
            console.log('solve for policy iteration ' + (++iteration));
            // console.log('policyValuesNew: ', policyValuesNew);
            if (iteration === 1000) {
                throw new Error('cannot converge after 1000 steps!');
            }
        } while (!this.valueConverged(policyValues, policyValuesNew))
        return policyValues;
    },
    policyValues: function (valuesOld, policy) {
        policyValuesNew = [];
        for (var i = 0; i < this.states.length; i++) {
            var state = this.states[i];
            var action = policy[i];
            var policyValue = this.qValue(valuesOld, state, action);
            policyValuesNew.push(policyValue);
        }
        return policyValuesNew;
    },
    qValue: function (valuesOld, state, action) {
        var value = 0;
        if (action) {
            var transitions = this.mdp.transitions(state, action);
            var targetStates = Object.keys(transitions);
            for (var i = 0; i < targetStates.length; i++) {
                var targetState = targetStates[i];
                var valueOld = valuesOld[this.states.indexOf(targetState)];
                value += transitions[targetState] * (this.mdp.reward(state, action, targetState) + this.gama * valueOld);
            }
        }
        return value;
    },
    converged: function (policyOld, policyCurr) {
        return m.arrayEq(policyOld, policyCurr);
    },
    valueConverged: function (valuesOld, valuesCurr) {
        return m.arrayEq(valuesOld, valuesCurr);
    },
    randomPolicy: function () {
        var policy = [];
        for (var i = 0; i < this.states.length; i++) {
            var state = this.states[i];
            var actions = this.mdp.availableActions(state);
            if (!actions.length) {
                policy.push(null);
            } else {
                policy.push(actions[m.randomInt(0, actions.length)]);
            }
        }
        return policy;
    }
}



var gameRouter = (new GameBuilder()).build();
gameRouter.interactor.start();


function Test() {
    var assert = {
        eq: function (curr, exp) {
            m.eq(curr, exp, function (_curr, _exp) {
                throw new Error('expect ' + _curr + ' (' + typeof(_curr) + ') to equal ' + _exp + ' (' + typeof(_exp) + ')');
            });
        },
        raise: function (func) {
            try {
                func();
            } catch (e) {
                return
            }
            throw new Error('expect an error to raise, but no error');
        },
        arrayEq: function (curr, exp) {
            m.arrayEq(curr, exp, function (_curr, _exp) {
                throw new Error('expect ' + _curr + ' to equal ' + _exp);
            });
        },
        objEq: function(curr, exp) {
            m.objEq(curr, exp, function (_curr, _exp) {
                throw new Error('expect ' + _curr + ' (' + typeof(_curr) + ') to equal ' + _exp + ' (' + typeof(_exp) + ')');
            });
        }
    }
    var all = {
        testAssert: function () {
            assert.raise(function () { assert.eq('', null) });
            assert.raise(function () { assert.arrayEq([], [1, 2]) });
            assert.raise(function () { assert.objEq({a:1}, {a:2}) });
        },
        testIsArray: function () {
            assert.eq(m.isArray([]), true);
            assert.eq(m.isArray([1]), true);
            assert.eq(m.isArray(''), false);
            assert.eq(m.isArray(undefined), false);
            assert.eq(m.isArray(1), false);
        },
        testArgMax: function() {
            assert.eq(m.argmax([1, 2, 3]), 2);
            assert.eq(m.argmax({1: 1, 2: 3, 3: 2}), '2');
        },
        testZeroArray: function () {
            assert.arrayEq(m.zeroArray([2,1]), [[0], [0]]);
            assert.arrayEq(m.zeroArray([2,2]), [[0,0], [0,0]]);
            assert.arrayEq(m.zeroArray([]), 0);
            assert.arrayEq(m.zeroArray([1]), [0]);
        },
        testMDP: function () {
            var c = function (type, arg1, arg2) {
                if (type.isEnd()) {
                    return new GridCell(type, { reward: arg1 });
                } else if (type.isRoom()) {
                    return new GridCell(type, { value: arg1, qValue: arg2 });
                } else {
                    return new GridCell(type);
                }
            }
            var gridTable = new GridTable([
                [c(CellType.room), c(CellType.room), c(CellType.room), c(CellType.success, 10)],
                [c(CellType.room), c(CellType.wall), c(CellType.room), c(CellType.fail, -10)]
            ]);
            var noop = function () {}, observableNoop = function () { return {subscribe: noop} };
            var view = {initTable: noop, updateTable: noop, startValueIteration: observableNoop, startPolicyIteration: observableNoop, finishIteration: noop};
            var interactor = new GridTableInteractor(view, gridTable);
            var mdp = interactor.mdp;
            assert.arrayEq(mdp.states, ['0-0', '0-1', '0-2', '0-3', '1-0', '1-1', '1-2', '1-3']);
            assert.arrayEq(mdp.availableActions('0-2'), ['t', 'l', 'r', 'b']);
            assert.arrayEq(mdp.availableActions('1-1'), []);
            assert.arrayEq(mdp.availableActions('1-3'), []);
            assert.objEq(mdp.transitions('0-1', 'b'), {'0-0': 0.15, '0-1': 0.7, '0-2': 0.15});
            assert.objEq(mdp.transitions('0-1', 'r'), {'0-0': 0.15, '0-1': 0.15, '0-2': 0.7});
            assert.objEq(mdp.transitions('0-1', 't'), {'0-0': 0.15, '0-1': 0.7, '0-2': 0.15});
            assert.objEq(mdp.transitions('0-2', 'r'), {'0-1': 0.1, '0-2': 0.1, '0-3': 0.7, '1-2': 0.1});
            assert.objEq(mdp.transitions('0-2', 't'), {'0-1': 0.1, '0-2': 0.7, '0-3': 0.1, '1-2': 0.1});
            assert.objEq(mdp.transitions('0-2', 't'), {'0-1': 0.1, '0-2': 0.7, '0-3': 0.1, '1-2': 0.1});
            assert.raise(function () { mdp.transitions('0-3', 't'); });
            assert.raise(function () { mdp.transitions('1-3', 'l'); });
        },
        testMDPVISolver: function () {
            var mdp = {
                states: ['a', 'b', 't'],
                availableActions: function (state) {
                    return {
                        'a': ['a1', 'a2'],
                        'b': ['e'],
                        't': []
                    }[state];
                },
                reward: function (state, action, targetState) {
                    // console.log('reward for: ', state + '-' + action + '-' + targetState)
                    return {
                        'a-a1-a': 0, 'a-a1-b': 1, 'a-a2-a': 0, 'a-a2-b': 1,
                        'b-e-t': 1
                    }[state + '-' + action + '-' + targetState];
                },
                transitions: function (state, action) {
                    return {
                        'a-a1': {a: 0.7, b: .3}, 'a-a2': {a: 0.2, b: 0.8},
                        'b-e': {t: 1}
                    }[state + '-' + action];
                }
            };
            var solver = new MDPVISolver(mdp);
            assert.eq(solver.converged([1, 3], [1 + 1e-8, 3]), true);
            assert.eq(solver.converged([1, 3], [1 + 1e-6, 3]), false);
            assert.eq(solver.value({1: 3, 2: 2, 3: 5, 4: -1}), 5);
            assert.eq(solver.value({1: 3, 2: 2, 3: 5, 4: -1}), 5);
            assert.eq(solver.qValue([0, 1, 3], 'a', 'a1'), 0.7 * (0 + solver.gama * 0) + 0.3 * (1 + solver.gama * 1));
            assert.objEq(solver.qValues([0, 1, 3], 'a'), {
                a1: 0.7 * (0 + solver.gama * 0) + 0.3 * (1 + solver.gama * 1),
                a2: 0.2 * (0 + solver.gama * 0) + 0.8 * (1 + solver.gama * 1)
            });
            var policy = solver.solve();
            assert.eq(policy.nextAction('a'), 'a2');
            assert.eq(policy.nextAction('b'), 'e');
        },
        testMDPPISolver: function () {
            var mdp = {
                states: ['a', 'b', 't'],
                availableActions: function (state) {
                    return {
                        'a': ['a1', 'a2'],
                        'b': ['e'],
                        't': []
                    }[state];
                },
                reward: function (state, action, targetState) {
                    // console.log('reward for: ', state + '-' + action + '-' + targetState)
                    return {
                        'a-a1-a': 0, 'a-a1-b': 1, 'a-a2-a': 0, 'a-a2-b': 1,
                        'b-e-t': 1
                    }[state + '-' + action + '-' + targetState];
                },
                transitions: function (state, action) {
                    return {
                        'a-a1': {a: 0.7, b: .3}, 'a-a2': {a: 0.2, b: 0.8},
                        'b-e': {t: 1}
                    }[state + '-' + action];
                }
            };
            var solver = new MDPPISolver(mdp);
            var policy = solver.randomPolicy();
            assert.eq(policy.length, 3);
            assert.eq(policy[0] === 'a1' || policy[0] === 'a2', true);
            assert.eq(policy[1], 'e');
            assert.eq(solver.qValue([1, 1, 3], 'a', 'a1'), 0.7 * (0 + solver.gama * 1) + 0.3 * (1 + solver.gama * 1));
            assert.eq(solver.qValue([1, 4, 3], 'b', 'e'), 1 * (1 + solver.gama * 3));
            assert.arrayEq(solver.policyValues([1, 1, 3], ['a1', 'e', null]), [
                0.7 * (0 + solver.gama * 1) + 0.3 * (1 + solver.gama * 1), 1 * (1 + solver.gama * 3), 0
            ]);
            var policy = solver.solve();
            assert.eq(policy.nextAction('a'), 'a2');
            assert.eq(policy.nextAction('b'), 'e');
            assert.eq(policy.nextAction('t'), null);
        }
    }

    this.run = function() {
        console.log('start testing.');
        var testCount = 0, passedCount = 0;
        for(var t in all) {
            if (all.hasOwnProperty(t)) {
                testCount += 1;
                try {
                    all[t]();
                    console.log('.');
                    passedCount += 1;
                } catch (e) {
                    console.log(e);
                }
            }
        }
        console.log('testing end: ' + passedCount + '/' + testCount + ' passed.');
    }
}




(new Test()).run();




})(Crafty);