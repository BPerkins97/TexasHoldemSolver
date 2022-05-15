package icybee.solver.solver;

import com.alibaba.fastjson.JSONObject;
import icybee.solver.Card;
import icybee.solver.Deck;
import icybee.solver.GameTree;
import icybee.solver.RiverRangeManager;
import icybee.solver.compairer.Compairer;
import icybee.solver.nodes.*;
import icybee.solver.ranges.PrivateCards;
import icybee.solver.ranges.PrivateCardsManager;
import icybee.solver.ranges.RiverCombs;
import icybee.solver.trainable.DiscountedCfrTrainable;
import icybee.solver.trainable.Trainable;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by huangxuefeng on 2019/10/11.
 * contains code for cfr solver
 */
public class ParallelCfrPlusSolver extends Solver {
    public static final int PLAYER_FIRST_UNSURE = 0;
    public static final int PLAYER_SECOND_UNSURE = 1;
    public static final int IN_POSITION_PLAYER = 0;
    public static final int OUT_OF_POSITION_PLAYER = 1;
    PrivateCards[][] ranges;
    PrivateCards[] inPositionPlayersRange;
    PrivateCards[] outOfPositionPlayersRange;
    int[] initial_board;
    long initial_board_long;
    Compairer compairer;

    Deck deck;
    ForkJoinPool forkJoinPool;
    RiverRangeManager riverRangeManager;
    int player_number;
    int iterationNumber;
    PrivateCardsManager privateCardsManager;
    boolean debug;
    int print_interval;
    String logfile;
    Class<?> trainer;
    int[] round_deal; // TODO what is round deal? Prob a bad name
    int nthreads;
    double forkprob_action;
    double forkprob_chance;
    int fork_every_n_depth;
    int no_fork_subtree_size;

    MonteCarloAlg monteCarloAlg;

    public ParallelCfrPlusSolver(
            GameTree tree,
            PrivateCards[] inPositionPlayersRange,
            PrivateCards[] outOfPositionPlayersRange,
            int[] initial_board,
            Compairer compairer,
            Deck deck,
            int iteration_number,
            boolean debug,
            int print_interval,
            String logfile,
            Class<?> trainer,
            MonteCarloAlg monteCarloAlg,
            int nthreads,
            double forkprob_action,
            double forkprob_chance,
            int fork_between,
            int no_fork_subtree_size
    ) {
        super(tree);
        //if(board.length != 5) throw new RuntimeException(String.format("board length %d",board.length));
        this.initial_board = initial_board;
        this.initial_board_long = Card.boardInts2long(initial_board);
        this.logfile = logfile;
        this.trainer = trainer;

        inPositionPlayersRange = this.noDuplicateRange(inPositionPlayersRange, initial_board_long);
        outOfPositionPlayersRange = this.noDuplicateRange(outOfPositionPlayersRange, initial_board_long);

        this.inPositionPlayersRange = inPositionPlayersRange;
        this.outOfPositionPlayersRange = outOfPositionPlayersRange;
        this.player_number = 2;
        this.ranges = new PrivateCards[this.player_number][];
        this.ranges[IN_POSITION_PLAYER] = inPositionPlayersRange;
        this.ranges[OUT_OF_POSITION_PLAYER] = outOfPositionPlayersRange;
        this.compairer = compairer;

        this.deck = deck;

        this.riverRangeManager = new RiverRangeManager(compairer);
        this.iterationNumber = iteration_number;

        PrivateCards[][] private_cards = new PrivateCards[this.player_number][];
        private_cards[IN_POSITION_PLAYER] = inPositionPlayersRange;
        private_cards[OUT_OF_POSITION_PLAYER] = outOfPositionPlayersRange;
        privateCardsManager = new PrivateCardsManager(private_cards, this.player_number, Card.boardInts2long(this.initial_board));
        this.debug = debug;
        this.print_interval = print_interval;
        this.monteCarloAlg = monteCarloAlg;
        if (nthreads >= 1) {
            this.nthreads = nthreads;
        } else if (nthreads == -1) {
            this.nthreads = Runtime.getRuntime().availableProcessors();
        } else {
            throw new RuntimeException("nthread not correct");
        }

        this.forkJoinPool = new ForkJoinPool(this.nthreads);
        if (forkprob_action > 1 || forkprob_action < 0)
            throw new RuntimeException(String.format("forkprob action not between [0,1] : %s", forkprob_action));
        if (forkprob_chance > 1 || forkprob_chance < 0)
            throw new RuntimeException(String.format("forkprob chance not between [0,1] : %s", forkprob_chance));
        this.forkprob_action = forkprob_action;
        this.forkprob_chance = forkprob_chance;
        this.fork_every_n_depth = fork_between;
        this.no_fork_subtree_size = no_fork_subtree_size;
        System.out.println(String.format("Using %s threads", this.nthreads));
    }

    PrivateCards[] playerHands(int player) {
        if (player == 0) {
            return inPositionPlayersRange;
        } else if (player == 1) {
            return outOfPositionPlayersRange;
        } else {
            throw new RuntimeException("player not found");
        }
    }

    /**
     * Collects the probability of reaching this point for all startings hands.
     * Is only used for preflop.
     * @return Probability of each hand getting to this round.
     */
    private float[][] getReachProbs() {
        float[][] retval = new float[this.player_number][];
        for (int player = 0; player < this.player_number; player++) {
            PrivateCards[] player_cards = this.playerHands(player);
            float[] reach_prob = new float[player_cards.length];
            for (int i = 0; i < player_cards.length; i++) {
                reach_prob[i] = player_cards[i].weight;
            }
            retval[player] = reach_prob;
        }
        return retval;
    }

    public PrivateCards[] noDuplicateRange(PrivateCards[] private_range, long board_long) {
        List<PrivateCards> range_array = new ArrayList<>();
        Map<Integer, Boolean> rangekv = new HashMap<>();
        for (PrivateCards one_range : private_range) {
            if (one_range == null) throw new RuntimeException();
            if (rangekv.get(one_range.hashCode()) != null)
                throw new RuntimeException(String.format("duplicated key %d", one_range));
            rangekv.put(one_range.hashCode(), Boolean.TRUE);
            long hand_long = Card.boardInts2long(new int[]{
                    one_range.card1,
                    one_range.card2
            });
            if (!Card.boardsHasIntercept(hand_long, board_long)) {
                range_array.add(one_range);
            }
        }
        PrivateCards[] ret = new PrivateCards[range_array.size()];
        range_array.toArray(ret);
        return ret;
    }

    // We initialize the Game Tree with the algorithms we want (CFR+, Discounted CFR, etc.)
    void setTrainable(GameTreeNode root) throws NoSuchMethodException, InvocationTargetException, IllegalAccessException, InstantiationException {
        if (root instanceof ActionNode) {
            ActionNode action_node = (ActionNode) root;

            int player = action_node.getPlayer();
            PrivateCards[] player_privates = this.ranges[player];

            action_node.setTrainable((Trainable) this.trainer.getConstructor(ActionNode.class, PrivateCards[].class).newInstance(action_node, player_privates));

            List<GameTreeNode> childrens = action_node.getChildrens();
            for (GameTreeNode one_child : childrens) setTrainable(one_child);
        } else if (root instanceof ChanceNode) {
            ChanceNode chanceNode = (ChanceNode) root;
            List<GameTreeNode> childrens = chanceNode.getChildren();
            for (GameTreeNode one_child : childrens) setTrainable(one_child);
        } else if (root instanceof TerminalNode) {

        } else if (root instanceof ShowdownNode) {

        }

    }

    private Double getValue(Map<String, Object> meta, String key) {
        Object value = meta.get(key);
        if (value instanceof Integer) {
            return ((Integer) value).doubleValue();
        } else if (value instanceof Double) {
            return (Double) value;
        } else {
            return Double.valueOf(0);
        }

    }

    @Override
    public void train(Map training_config) throws Exception {
        setTrainable(tree.getRoot());

        PrivateCards[][] player_privates = new PrivateCards[this.player_number][];
        player_privates[IN_POSITION_PLAYER] = privateCardsManager.getPreflopCards(IN_POSITION_PLAYER);
        player_privates[OUT_OF_POSITION_PLAYER] = privateCardsManager.getPreflopCards(OUT_OF_POSITION_PLAYER);

        // TODO Figure out how this works later. It is only used for determining when to leave the training loop.
        BestResponse br = new BestResponse(player_privates, this.player_number, this.compairer, this.privateCardsManager, this.riverRangeManager, this.deck, this.debug);

        br.printExploitability(tree.getRoot(), 0, tree.getRoot().getPot().floatValue(), initial_board_long);

        // Load the initial probabilies of cards that see this round
        float[][] reach_probs = this.getReachProbs();

        FileWriter fileWriter = null;
        if (this.logfile != null) fileWriter = new FileWriter(this.logfile);

        long begintime = System.currentTimeMillis();
        long endtime = System.currentTimeMillis();

        Double stopAtExploitability = this.getValue(training_config, "stopAtExploitability");
        for (int i = 0; i < this.iterationNumber; i++) {
            for (int player_id = 0; player_id < this.player_number; player_id++) {
                if (this.debug) {
                    System.out.println(String.format(
                            "---------------------------------     player %s --------------------------------",
                            player_id
                    ));
                }
                this.round_deal = new int[]{-1, -1, -1, -1}; // I don't know about this

                // Create a new CFR Task for each iteration and each player
                // This does the heavy lifting
                // I don't know about using the this as solverEnvironment as this is only a reference and might or maybe we don't have this issue because some magic
                CfrTask task = new CfrTask(player_id, this.tree.getRoot(), reach_probs, i, this.initial_board_long, this);
                forkJoinPool.invoke(task);
            }

            // Does nothing special, just logging and early exit
            if (i % this.print_interval == 0) {
                float exploitability = br.printExploitability(tree.getRoot(), i + 1, tree.getRoot().getPot().floatValue(), initial_board_long);
                printDebugInformation(fileWriter, begintime, i, exploitability);
                if (stopAtExploitability > exploitability) {
                    break;
                }
            }
        }

        // More logging
        endtime = System.currentTimeMillis();
        long time_ms = endtime - begintime;
        System.out.println("++++++++++++++++");
        System.out.println(String.format("solve finish, total time used: %.2fs", (float) time_ms / 1000));
        if (this.logfile != null) {
            fileWriter.flush();
            fileWriter.close();
        }
        forkJoinPool.shutdown();
    }

    private void printDebugInformation(FileWriter fileWriter, long begintime, int i, float exploitability) throws IOException {
        long endtime;
        endtime = System.currentTimeMillis();
        long time_ms = endtime - begintime;
        System.out.println(String.format("time used: %.2fs", (float) time_ms / 1000));
        System.out.println("-------------------");
        if (this.logfile != null) {
            JSONObject jo = new JSONObject();
            jo.put("iteration", i);
            jo.put("exploitibility", exploitability);
            jo.put("time_ms", time_ms);
            if (this.logfile != null) fileWriter.write(String.format("%s\n", jo.toJSONString()));
        }
    }

    class CfrTask extends RecursiveTask<float[]> {
        int player;
        GameTreeNode node;
        float[][] reach_probs;
        int iter;
        long current_board;
        ParallelCfrPlusSolver solver_env;

        public CfrTask(int player, GameTreeNode node, float[][] reach_probs, int iter, long current_board, ParallelCfrPlusSolver solver_env) {
            this.player = player;
            this.node = node;
            this.reach_probs = reach_probs;
            this.iter = iter;
            this.current_board = current_board;
            this.solver_env = solver_env;
        }

        @Override
        protected float[] compute() {
            return this.cfr(this.player, this.node, this.reach_probs, this.iter, this.current_board);
        }

        /**
         * Calculate the value of being at this node
         * @param player
         * @param node
         * @param reach_probs
         * @param iter
         * @param current_board
         * @return
         */
        float[] cfr(int player, GameTreeNode node, float[][] reach_probs, int iter, long current_board) {
            // What is the difference between showdown and terminal ?
            // Showdown means both player got to the river, terminal means all others gave up
            switch (node.getType()) {
                case ACTION:
                    return actionUtility(player, (ActionNode) node, reach_probs, iter, current_board);
                case SHOWDOWN:
                    return showdownUtility(player, (ShowdownNode) node, reach_probs, iter, current_board);
                case TERMINAL:
                    return terminalUtility(player, (TerminalNode) node, reach_probs, iter, current_board);
                case CHANCE:
                    return chanceUtility(player, (ChanceNode) node, reach_probs, iter, current_board);
                default:
                    throw new RuntimeException("node type unknown");
            }
        }

        /**
         * Computes the EV of each dealt card for the player
         * @param player
         * @param node
         * @param reach_probs
         * @param iter
         * @param current_board
         * @return
         */
        float[] chanceUtility(int player, ChanceNode node, float[][] reach_probs, int iter, long current_board) {
            List<Card> cards = this.solver_env.deck.getCards();
            if (cards.size() != node.getChildren().size()) throw new RuntimeException();

            int card_num = node.getCards().size();
            // Total amount of cards - the cards on board - my preflop cards
            int possible_deals = node.getChildren().size() - Card.long2board(current_board).length - 2;

            float[] chance_utility = new float[reach_probs[player].length];
            int random_deal = 0, cardcount = 0;
            if (this.solver_env.monteCarloAlg == MonteCarloAlg.PUBLIC) {
                if (this.solver_env.round_deal[GameTreeNode.gameRound2int(node.getRound())] == -1) {
                    random_deal = ThreadLocalRandom.current().nextInt(1, possible_deals + 1 + 2);
                    this.solver_env.round_deal[GameTreeNode.gameRound2int(node.getRound())] = random_deal;
                } else {
                    random_deal = this.solver_env.round_deal[GameTreeNode.gameRound2int(node.getRound())];
                }
            }
            CfrTask[] tasklist = new CfrTask[node.getCards().size()];
            boolean forkAt = false;
            if (this.solver_env.forkprob_chance == 1) {
                forkAt = true;
            } else if (this.solver_env.forkprob_chance == 0) {
                forkAt = false;
            } else if (Math.random() < this.solver_env.forkprob_chance) {
                forkAt = true;
            }

            if (node.depth % this.solver_env.fork_every_n_depth != 0
                    || node.subtree_size <= this.solver_env.no_fork_subtree_size) forkAt = false;

            for (int card = 0; card < node.getCards().size(); card++) {
                GameTreeNode one_child = node.getChildren().get(card);
                Card one_card = node.getCards().get(card);
                long card_long = Card.boardCards2long(new Card[]{one_card});

                if (Card.boardsHasIntercept(card_long, current_board)) continue;
                cardcount += 1;

                if (one_child == null || one_card == null) throw new RuntimeException("child is null");

                long new_board_long = current_board | card_long;
                if (this.solver_env.monteCarloAlg == MonteCarloAlg.PUBLIC) {
                    if (cardcount == random_deal) {
                        // crete job
                        CfrTask task = new CfrTask(this.player, one_child, reach_probs, iter, new_board_long, this.solver_env);
                        //task.fork();
                        return task.compute();
                    } else {
                        continue;
                    }
                }

                PrivateCards[] playerPrivateCard = this.solver_env.ranges[player];
                PrivateCards[] oppoPrivateCards = this.solver_env.ranges[1 - player];

                float[][] new_reach_probs = new float[2][];

                new_reach_probs[player] = new float[playerPrivateCard.length];
                new_reach_probs[1 - player] = new float[oppoPrivateCards.length];

                if (playerPrivateCard.length != reach_probs[player].length)
                    throw new RuntimeException("length not match");
                if (oppoPrivateCards.length != reach_probs[1 - player].length)
                    throw new RuntimeException("length not match");

                for (int one_player = 0; one_player < 2; one_player++) {
                    int player_hand_len = this.solver_env.ranges[one_player].length;
                    for (int player_hand = 0; player_hand < player_hand_len; player_hand++) {
                        PrivateCards one_private = this.solver_env.ranges[one_player][player_hand];
                        long privateBoardLong = one_private.toBoardLong();
                        if (Card.boardsHasIntercept(card_long, privateBoardLong)) continue;
                        new_reach_probs[one_player][player_hand] = reach_probs[one_player][player_hand] / possible_deals; // TODO why do we divide here?
                    }
                }

                if (Card.boardsHasIntercept(current_board, card_long))
                    throw new RuntimeException("board has intercept with dealt card");

                CfrTask task = new CfrTask(this.player, one_child, new_reach_probs, iter, new_board_long, this.solver_env);
                if (forkAt) task.fork();
                tasklist[card] = task;
            }

            for (int card = 0; card < node.getCards().size(); card++) {
                CfrTask task = tasklist[card];
                if (task == null) continue;
                float[] child_utility;
                if (forkAt) {
                    child_utility = task.join();
                } else {
                    child_utility = task.compute();
                }
                if (child_utility.length != chance_utility.length) throw new RuntimeException("length not match");
                for (int i = 0; i < child_utility.length; i++)
                    chance_utility[i] += child_utility[i];
            }

            if (this.solver_env.monteCarloAlg == MonteCarloAlg.PUBLIC) {
                throw new RuntimeException("not possible");
            }
            return chance_utility;
        }

        /**
         * Computes the EV of each action for the player
         * @param player
         * @param node
         * @param reach_probs
         * @param iter
         * @param current_board
         * @return
         */
        float[] actionUtility(int player, ActionNode node, float[][] reach_probs, int iter, long current_board) {
            int oppo = 1 - player;
            PrivateCards[] node_player_private_cards = this.solver_env.ranges[node.getPlayer()];
            Trainable trainable = node.getTrainable();

            float[] payoffs = new float[this.solver_env.ranges[player].length];
            List<GameTreeNode> children = node.getChildrens();
            List<GameActions> actions = node.getActions();

            boolean forkAt = false;
            if (this.solver_env.forkprob_action == 1) {
                forkAt = true;
            } else if (this.solver_env.forkprob_action == 0) {
                forkAt = false;
            } else if (Math.random() < this.solver_env.forkprob_action) {
                forkAt = true;
            }

            if (node.depth % this.solver_env.fork_every_n_depth != 0
                    || node.subtree_size <= this.solver_env.no_fork_subtree_size) forkAt = false;

            float[] current_strategy = trainable.getcurrentStrategy();
            if (this.solver_env.debug) {
                for (float one_strategy : current_strategy) {
                    if (one_strategy != one_strategy) {
                        System.out.println(Arrays.toString(current_strategy));
                        throw new RuntimeException();
                    }

                }
                for (int one_player = 0; one_player < this.solver_env.player_number; one_player++) {
                    float[] one_reach_prob = reach_probs[one_player];
                    for (float one_prob : one_reach_prob) {
                        if (one_prob != one_prob)
                            throw new RuntimeException();
                    }
                }
            }
            if (current_strategy.length != actions.size() * node_player_private_cards.length) {
                node.printHistory();
                throw new RuntimeException(String.format(
                        "length not match %s - %s \n action size %s private_card size %s"
                        , current_strategy.length
                        , actions.size() * node_player_private_cards.length
                        , actions.size()
                        , node_player_private_cards.length
                ));
            }

            float[] regrets = new float[actions.size() * node_player_private_cards.length];

            float[][] all_action_utility = new float[actions.size()][];
            int node_player = node.getPlayer();

            CfrTask[] tasklist = new CfrTask[actions.size()];

            for (int action_id = 0; action_id < actions.size(); action_id++) {
                float[][] new_reach_prob = new float[this.solver_env.player_number][];
                new_reach_prob[1 - node_player] = reach_probs[1 - node_player];
                float[] player_new_reach = new float[reach_probs[node_player].length];
                for (int hand_id = 0; hand_id < player_new_reach.length; hand_id++) {
                    float strategy_prob = current_strategy[hand_id + action_id * node_player_private_cards.length];
                    player_new_reach[hand_id] = reach_probs[node_player][hand_id] * strategy_prob;
                }
                new_reach_prob[node_player] = player_new_reach;

                CfrTask task = new CfrTask(this.player, children.get(action_id), new_reach_prob, iter, current_board, this.solver_env);

                if (forkAt) {
                    task.fork();
                }
                tasklist[action_id] = task;
            }

            for (int action_id = 0; action_id < actions.size(); action_id++) {
                CfrTask task = tasklist[action_id];
                if (task == null) continue;

                float[] action_utilities;
                if (forkAt) {
                    try {
                        action_utilities = task.join();
                    } catch (Exception e) {
                        throw new RuntimeException("future get error");
                    }
                } else {
                    action_utilities = task.compute();
                }
                all_action_utility[action_id] = action_utilities;

                if (action_utilities.length != payoffs.length) {
                    System.out.println("errmsg");
                    System.out.println(String.format("node player %s ", node.getPlayer()));
                    node.printHistory();
                    throw new RuntimeException(
                            String.format(
                                    "action and payoff length not match %s - %s"
                                    , action_utilities.length
                                    , payoffs.length
                            )
                    );
                }

                for (int hand_id = 0; hand_id < action_utilities.length; hand_id++) {
                    if (player == node.getPlayer()) {
                        float strategy_prob = current_strategy[hand_id + action_id * node_player_private_cards.length];
                        payoffs[hand_id] += strategy_prob * action_utilities[hand_id];
                    } else {
                        payoffs[hand_id] += action_utilities[hand_id];
                    }
                }
            }


            if (player == node.getPlayer()) {
                for (int i = 0; i < node_player_private_cards.length; i++) {
                    for (int action_id = 0; action_id < actions.size(); action_id++) {
                        regrets[action_id * node_player_private_cards.length + i] = all_action_utility[action_id][i] - payoffs[i];
                    }
                }
                trainable.updateRegrets(regrets, iter + 1, reach_probs[player]);
                if (trainable instanceof DiscountedCfrTrainable) {
                    DiscountedCfrTrainable dct = (DiscountedCfrTrainable) trainable;
                    dct.setEvs(payoffs);
                    dct.setReach_probs(reach_probs);
                }
            }

            return payoffs;
        }

        /**
         * Calculates the expected Value per combo.
         * @param player The Player for which to calculate the EV
         * @param node The Node
         * @param reachProbability The probability of the individual combos getting to showdown
         * @param iter Iteration, not used
         * @param current_board the hash value of the current board (I guess)
         * @return Expected Value by combo held
         */
        float[] showdownUtility(int player, ShowdownNode node, float[][] reachProbability, int iter, long current_board) {
            int oppo = player == IN_POSITION_PLAYER ? OUT_OF_POSITION_PLAYER : IN_POSITION_PLAYER;
            float win_payoff = node.get_payoffs(ShowdownNode.ShowDownResult.NO_TIE, player)[player].floatValue();
            float lose_payoff = node.get_payoffs(ShowdownNode.ShowDownResult.NO_TIE, oppo)[player].floatValue();
            PrivateCards[] player_private_cards = this.solver_env.ranges[player];
            PrivateCards[] oppo_private_cards = this.solver_env.ranges[oppo];

            RiverCombs[] player_combs = this.solver_env.riverRangeManager.getRiverCombos(player, player_private_cards, current_board);
            RiverCombs[] oppo_combs = this.solver_env.riverRangeManager.getRiverCombos(oppo, oppo_private_cards, current_board);

            float[] payoffs = new float[player_private_cards.length];


            float winsum = 0;
            float[] card_winsum = new float[52];


            if (this.solver_env.debug) {
                System.out.println("[PRESHOWDOWN]=======================");
                System.out.println(String.format("player0 reach_prob %s", Arrays.toString(reachProbability[0])));
                System.out.println(String.format("player1 reach_prob %s", Arrays.toString(reachProbability[1])));
                System.out.print("preflop combos: ");
                for (RiverCombs one_river_comb : player_combs) {
                    System.out.print(String.format("%s(%s) "
                            , one_river_comb.private_cards.toString()
                            , one_river_comb.rank
                    ));
                }
                System.out.println();
            }

            // Evaluate all hands vs all hands of opponent and add to the winsum
            int j = 0;
            for (int i = 0; i < player_combs.length; i++) {
                RiverCombs one_player_comb = player_combs[i];
                while (j < oppo_combs.length && one_player_comb.rank < oppo_combs[j].rank) {
                    RiverCombs one_oppo_comb = oppo_combs[j];
                    winsum += reachProbability[oppo][one_oppo_comb.preflopComboIndex];
                    if (this.solver_env.debug) {
                        if (one_player_comb.preflopComboIndex == 0) {
                            System.out.print(String.format("[%s]%s:%s-%s(%s) "
                                    , j
                                    , one_oppo_comb.private_cards.toString()
                                    , this.solver_env.ranges[oppo][one_oppo_comb.preflopComboIndex].weight
                                    , winsum
                                    , one_oppo_comb.rank
                            ));
                        }
                    }

                    card_winsum[one_oppo_comb.private_cards.card1] += reachProbability[oppo][one_oppo_comb.preflopComboIndex];
                    card_winsum[one_oppo_comb.private_cards.card2] += reachProbability[oppo][one_oppo_comb.preflopComboIndex];
                    j++;
                }
                if (this.solver_env.debug) {
                    //调查这里为什么加完了是负数
                    System.out.println(String.format("Before Adding %s, win_payoff %s winsum %s, subcard1 %s subcard2 %s"
                            , payoffs[one_player_comb.preflopComboIndex]
                            , win_payoff
                            , winsum
                            , -card_winsum[one_player_comb.private_cards.card1]
                            , -card_winsum[one_player_comb.private_cards.card2]
                    ));
                }
                payoffs[one_player_comb.preflopComboIndex] = (winsum
                        - card_winsum[one_player_comb.private_cards.card1]
                        - card_winsum[one_player_comb.private_cards.card2]
                ) * win_payoff;
                if (this.solver_env.debug) {
                    if (one_player_comb.preflopComboIndex == 0) {
                        System.out.println(String.format("winsum %s", winsum));
                    }
                }
            }

            // Evaluate all possible combinations against opponents combinations and add the losing ones
            // Why do we save the win/loss for each card individually?
            float summedLoseProbabilty = 0;
            float[] perCardProbabilty = new float[52]; // This is used to calculate the blocker effect
            for (int i = 0; i < perCardProbabilty.length; i++) perCardProbabilty[i] = 0;

            j = oppo_combs.length - 1;
            for (int i = player_combs.length - 1; i >= 0; i--) {
                RiverCombs one_player_comb = player_combs[i];

                // Save the probability of all the opponents winning combos that reached the river
                while (j >= 0 && one_player_comb.rank > oppo_combs[j].rank) {
                    RiverCombs one_oppo_comb = oppo_combs[j];
                    summedLoseProbabilty += reachProbability[oppo][one_oppo_comb.preflopComboIndex];
                    if (this.solver_env.debug) {
                        if (one_player_comb.preflopComboIndex == 0) {
                            System.out.print(String.format("lose %s:%s "
                                    , one_oppo_comb.private_cards.toString()
                                    , this.solver_env.ranges[oppo][one_oppo_comb.preflopComboIndex].weight
                            ));
                        }
                    }

                    perCardProbabilty[one_oppo_comb.private_cards.card1] += reachProbability[oppo][one_oppo_comb.preflopComboIndex];
                    perCardProbabilty[one_oppo_comb.private_cards.card2] += reachProbability[oppo][one_oppo_comb.preflopComboIndex];
                    j--;
                }
                if (this.solver_env.debug) {
                    System.out.println(String.format("Before Substract %s", payoffs[one_player_comb.preflopComboIndex]));
                }
                // What does this do?
                // We save the payoffs of all the preflop combos
                // Theory: We calculate the payoff like this the probabilty the opponent has any combo that beats us minus the proability of us having blockers
                payoffs[one_player_comb.preflopComboIndex] += (summedLoseProbabilty
                        - perCardProbabilty[one_player_comb.private_cards.card1]
                        - perCardProbabilty[one_player_comb.private_cards.card2]
                ) * lose_payoff;
                if (this.solver_env.debug) {
                    if (one_player_comb.preflopComboIndex == 0) {
                        System.out.println(String.format("losssum %s", summedLoseProbabilty));
                    }
                }
            }
            if (this.solver_env.debug) {
                System.out.println();
                System.out.println("[SHOWDOWN]============");
                node.printHistory();
                System.out.println(String.format("loss payoffs: %s", lose_payoff));
                System.out.println(String.format("oppo sum %s, substracted payoff %s", summedLoseProbabilty, payoffs[0]));
            }

            return payoffs;
        }

        /**
         * Calculates the EV of the Terminal Node
         * @param player
         * @param node
         * @param reach_prob
         * @param iter
         * @param current_board
         * @return
         */
        float[] terminalUtility(int player, TerminalNode node, float[][] reach_prob, int iter, long current_board) {

            Double player_payoff = node.get_payoffs()[player];
            if (player_payoff == null)
                throw new RuntimeException(String.format("player %d 's payoff is not found", player));

            int oppo = 1 - player;
            PrivateCards[] player_hand = playerHands(player);
            PrivateCards[] oppo_hand = playerHands(oppo);

            float[] payoffs = new float[this.solver_env.playerHands(player).length];

            float oppo_sum = 0;
            float[] oppo_card_sum = new float[52];
            Arrays.fill(oppo_card_sum, 0);

            // Sums the probabilty that the opponent folded
            for (int i = 0; i < oppo_hand.length; i++) {
                oppo_card_sum[oppo_hand[i].card1] += reach_prob[oppo][i];
                oppo_card_sum[oppo_hand[i].card2] += reach_prob[oppo][i];
                oppo_sum += reach_prob[oppo][i];
            }

            if (this.solver_env.debug) {
                System.out.println("[PRETERMINAL]============");
            }
            for (int i = 0; i < player_hand.length; i++) {
                PrivateCards one_player_hand = player_hand[i];
                if (Card.boardsHasIntercept(current_board, Card.boardInts2long(new int[]{one_player_hand.card1, one_player_hand.card2}))) {
                    continue;
                }
                Integer oppo_same_card_ind = this.solver_env.privateCardsManager.indPlayer2Player(player, oppo, i);
                float plus_reach_prob;
                if (oppo_same_card_ind == null) {
                    plus_reach_prob = 0;
                } else {
                    plus_reach_prob = reach_prob[oppo][oppo_same_card_ind];
                }
                // Calculates the payoffs taking into account blocker effects
                payoffs[i] = player_payoff.floatValue() * (
                        oppo_sum - oppo_card_sum[one_player_hand.card1]
                                - oppo_card_sum[one_player_hand.card2]
                                + plus_reach_prob
                );
                if (this.solver_env.debug) {
                    System.out.println(String.format("oppo_card_sum1 %s ", oppo_card_sum[one_player_hand.card1]));
                    System.out.println(String.format("oppo_card_sum2 %s ", oppo_card_sum[one_player_hand.card2]));
                    System.out.println(String.format("reach_prob i %s ", plus_reach_prob));
                }
            }

            if (this.solver_env.debug) {
                System.out.println("[TERMINAL]============");
                node.printHistory();
                System.out.println(String.format("PPPayoffs: %s", player_payoff));
                System.out.println(String.format("reach prob %s", reach_prob[oppo][0]));
                System.out.println(String.format("oppo sum %s, substracted sum %s", oppo_sum, payoffs[0] / player_payoff));
                System.out.println(String.format("substracted sum %s", payoffs[0]));
            }
            return payoffs;
        }

    }

}
