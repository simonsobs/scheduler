import yaml
import os.path as op
from dataclasses import dataclass
import datetime as dt
from typing import List

from . import basic
from .. import config as cfg, core, utils, source as src, rules as ru, commands as cmd, instrument as inst


@dataclass(frozen=True)
class FlexPolicy(basic.BasePolicy):
    """a flexible policy. `config` is a string yaml config *content*"""
    config_text: str
    rules: List[core.Rule]
    post_rules: List[core.Rule]
    merge_order: List[str]

    @staticmethod
    def rule_constructor(loader, node):
        """loader for rule"""
        kwargs = loader.construct_mapping(node)
        rule_name = kwargs.pop('name')
        constraint = kwargs.pop('constraint', None)

        # some rules that require randomization
        if rule_name in ['make-source-scan', 'rephase-first']:
            today = dt.datetime.now()
            rng_key = utils.PRNGKey((today.year, today.month, today.day, kwargs.pop('seed', 0)))
            kwargs['rng_key'] = rng_key
        rule = ru.make_rule(rule_name, **kwargs)

        # if a constraint is specified, make a constrained rule instead.
        if constraint is not None:
            return ru.ConstrainedRule(rule, constraint)
        else:
            return rule

    @classmethod
    def from_config(cls, config: str):
        """populate policy object from a yaml config file"""
        # first we load the text content of config into a string for
        # later use
        if op.isfile(config):
            with open(config, "r") as f:
                config_text = f.read()
        else:
            config_text = config
        # then we pre-load the config to populate some common fields in
        # the policy
        loader = cfg.get_loader()
        loader.add_constructor("!rule", cls.rule_constructor)
        config = yaml.load(config_text, Loader=loader)
        # remove fields that need special handling later on
        config.pop('blocks')
        # now we can construct the policy
        return cls(config_text=config_text, **config)

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        # prepare some specialized loaders: !source [source_name], !toast [toast schedule name]
        def source_constructor(t0, t1, loader, node):
            return src.source_gen_seq(loader.construct_scalar(node), t0, t1)
        def toast_constructor(loader, node):
            return utils.parse_sequence_from_toast(loader.construct_scalar(node))
        loader = cfg.get_loader()
        loader.add_constructor('!source', lambda loader, node: source_constructor(t0, t1, loader, node))
        loader.add_constructor('!toast', lambda loader, node: toast_constructor(loader, node))
        # load blocks for processing
        blocks = yaml.load(self.config_text, Loader=loader)["blocks"]
        return core.seq_trim(blocks, t0, t1)

    def transform(self, blocks: core.BlocksTree) -> core.BlocksTree:
        # apply each rule
        for rule in self.rules: 
            blocks = rule(blocks)

        return blocks

    def merge(self, blocks: core.BlocksTree) -> core.Blocks:
        """merge blocks into a single sequence by the order specified
        in self.merge_order, assuming an ascending priority order as moving
        down the merge_order list."""
        seq = None
        for query in self.merge_order:
            match, _ = core.seq_partition_with_query(query, blocks)
            if seq is None: 
                seq = match
                continue
            else:
                # match takes precedence
                seq = core.seq_merge(seq, match, flatten=True)

        # apply transformation if needed
        for rule in self.post_rules:
            seq = rule(seq)

        return core.seq_sort(seq)

    def block2cmd(self, block: core.Block):
        if isinstance(block, inst.ScanBlock):
            return cmd.CompositeCommand([
                    f"# {block.name}",
                    cmd.Goto(block.az, block.alt),
                    cmd.BiasDets(),
                    cmd.Wait(block.t0),
                    cmd.BiasStep(),
                    cmd.Scan(block.name, block.t1, block.throw),
                    cmd.BiasStep(),
                    "",
            ])

    def seq2cmd(self, seq: core.Blocks):
        """map a scan to a command"""
        commands = core.seq_flatten(core.seq_map(self.block2cmd, seq))
        commands = [cmd.Preamble()] + commands
        return cmd.CompositeCommand(commands)